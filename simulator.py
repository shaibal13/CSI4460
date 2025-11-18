#!/usr/bin/env python3
"""
NIDS (Network Intrusion Detection System) Simulator
Simulates real-time network traffic monitoring and attack detection using FT-Transformer model.
This simulates a live network monitoring scenario with real-time packet analysis.
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from collections import deque
import random

from data_loader import load_unsw_kaggle_dataset
from models.ft_transformer import FTTransformerTrainer
from utils.metrics import evaluate_binary


class NIDSSimulator:
    def __init__(self, model_path, device, batch_size=512, threshold=0.5):
        """
        Initialize NIDS Simulator with FT-Transformer model.
        
        Args:
            model_path: Path to trained FT-Transformer checkpoint
            device: torch device (cuda or cpu)
            batch_size: Batch size for inference
            threshold: Decision threshold for attack detection (default: 0.5)
        """
        self.device = device
        self.batch_size = batch_size
        self.threshold = threshold
        self.model_path = model_path
        self.model = None
        self.categories = None
        self.num_cont = None
        
        # Real-time monitoring statistics
        self.start_time = None
        self.total_packets = 0
        self.normal_packets = 0
        self.attack_packets = 0
        self.attack_alerts = []  # Store attack alerts with timestamps
        self.recent_predictions = deque(maxlen=100)  # Last 100 predictions for monitoring
        
        # Performance metrics
        self.processing_times = deque(maxlen=1000)
        
    def load_model(self, categories, num_cont, pos_weight_ratio=1.0):
        """Load the FT-Transformer model from checkpoint."""
        print(f"[NIDS] Loading FT-Transformer model from: {self.model_path}")
        
        self.categories = categories
        self.num_cont = num_cont
        
        trainer = FTTransformerTrainer(
            categories=categories,
            num_cont=num_cont,
            pos_weight_ratio=pos_weight_ratio,
            dim=128,
            depth=4,
            heads=8,
            dropout=0.1,
            lr=1e-4
        )
        trainer.load_best(self.model_path, self.device)
        self.model = trainer
        
        print("[NIDS] Model loaded successfully! System ready for monitoring.")
        
    def predict_single(self, x_num, x_cat):
        """
        Predict on a single network packet (for real-time simulation).
        
        Args:
            x_num: Single numeric feature vector (1D array)
            x_cat: Single categorical feature vector (1D array)
            
        Returns:
            prob: Attack probability (0-1)
            prediction: Binary prediction (0=normal, 1=attack)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Reshape to batch of 1
        x_num_batch = x_num.reshape(1, -1)
        x_cat_batch = x_cat.reshape(1, -1)
        
        start_time = time.time()
        prob = self.model.predict_proba(x_num_batch, x_cat_batch, device=self.device, batch_size=1)[0]
        processing_time = time.time() - start_time
        
        self.processing_times.append(processing_time)
        prediction = 1 if prob >= self.threshold else 0
        
        return prob, prediction
    
    def simulate_network_traffic(self, X_num, X_cat, y_true=None, 
                                  packet_rate=10, duration=None, 
                                  show_alerts=True, alert_cooldown=5):
        """
        Simulate real-time network traffic monitoring.
        Processes packets at a specified rate to simulate live network monitoring.
        
        Args:
            X_num: Numeric features array
            X_cat: Categorical features array
            y_true: True labels (optional, for evaluation)
            packet_rate: Packets per second to simulate
            duration: Maximum simulation duration in seconds (None = process all)
            show_alerts: Show attack alerts in real-time
            alert_cooldown: Minimum seconds between alert displays
            
        Returns:
            results: Dictionary with predictions and statistics
        """
        print("\n" + "="*80)
        print("  NIDS SIMULATOR")
        print("="*80)
        print(f"Model: FT-Transformer")
        print(f"Detection Threshold: {self.threshold}")
        print(f"Simulated Packet Rate: {packet_rate} packets/second")
        print(f"Total Packets Available: {len(X_num)}")
        if duration:
            print(f"Simulation Duration: {duration} seconds")
        print("="*80)
        print("\n[System] NIDS initialized and monitoring network traffic...")
        print("[System] Press Ctrl+C to stop monitoring\n")
        
        self.start_time = time.time()
        last_alert_time = {}
        all_probs = []
        all_preds = []
        
        # Calculate how many packets to process
        if duration:
            max_packets = int(packet_rate * duration)
            n_samples = min(max_packets, len(X_num))
        else:
            n_samples = len(X_num)
        
        # Shuffle to simulate random network traffic
        indices = np.random.permutation(len(X_num))[:n_samples]
        
        try:
            for idx, packet_idx in enumerate(indices):
                # Get packet data
                x_num = X_num[packet_idx]
                x_cat = X_cat[packet_idx]
                
                # Simulate packet arrival time
                packet_arrival = time.time()
                
                # Make prediction
                prob, pred = self.predict_single(x_num, x_cat)
                all_probs.append(prob)
                all_preds.append(pred)
                
                # Update statistics
                self.total_packets += 1
                if pred == 0:
                    self.normal_packets += 1
                else:
                    self.attack_packets += 1
                    
                    # Store attack alert
                    alert_info = {
                        'packet_id': packet_idx,
                        'timestamp': datetime.now(),
                        'probability': prob,
                        'true_label': y_true[packet_idx] if y_true is not None else None
                    }
                    self.attack_alerts.append(alert_info)
                    self.recent_predictions.append({
                        'packet_id': packet_idx,
                        'probability': prob,
                        'prediction': pred,
                        'timestamp': datetime.now()
                    })
                    
                    # Show alert if enabled and cooldown passed
                    if show_alerts:
                        current_time = time.time()
                        if packet_idx not in last_alert_time or \
                           (current_time - last_alert_time[packet_idx]) >= alert_cooldown:
                            true_info = ""
                            if y_true is not None:
                                is_correct = "✓" if pred == y_true[packet_idx] else "✗"
                                true_label = 'ATTACK' if y_true[packet_idx] == 1 else 'NORMAL'
                                true_info = f" | True: {true_label} {is_correct}"
                            
                            print(f" [ALERT] Packet #{packet_idx:6d} | "
                                  f"Attack Detected! | Confidence: {prob:.4f}{true_info}")
                            last_alert_time[packet_idx] = current_time
                
                # Store all predictions for validation
                self.recent_predictions.append({
                    'packet_id': packet_idx,
                    'probability': prob,
                    'prediction': pred,
                    'true_label': y_true[packet_idx] if y_true is not None else None,
                    'timestamp': datetime.now()
                })
                
                # Show normal packets occasionally (every N packets) for validation
                if pred == 0 and (idx + 1) % (packet_rate * 5) == 0:  # Show normal every 5 seconds
                    true_info = ""
                    if y_true is not None:
                        is_correct = "✓" if pred == y_true[packet_idx] else "✗"
                        true_label = 'ATTACK' if y_true[packet_idx] == 1 else 'NORMAL'
                        true_info = f" | True: {true_label} {is_correct}"
                    print(f" [INFO] Packet #{packet_idx:6d} | Normal Traffic | Confidence: {prob:.4f}{true_info}")
                
                # Show periodic status update
                if (idx + 1) % packet_rate == 0:  # Every second (if packet_rate packets processed)
                    elapsed = time.time() - self.start_time
                    attack_rate = (self.attack_packets / self.total_packets * 100) if self.total_packets > 0 else 0
                    normal_rate = (self.normal_packets / self.total_packets * 100) if self.total_packets > 0 else 0
                    avg_processing = np.mean(self.processing_times) * 1000 if self.processing_times else 0
                    
                    # Calculate accuracy if labels available
                    accuracy_info = ""
                    if y_true is not None and len(all_preds) > 0:
                        y_subset = y_true[indices[:len(all_preds)]]
                        correct = np.sum(np.array(all_preds) == y_subset)
                        accuracy = correct / len(all_preds)
                        accuracy_info = f" | Accuracy: {accuracy:.2%}"
                    
                    status_line = (
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Packets: {self.total_packets} | "
                        f"Normal: {self.normal_packets} ({normal_rate:.1f}%) | "
                        f"Attacks: {self.attack_packets} ({attack_rate:.1f}%) | "
                        f"Processing: {avg_processing:.1f}ms{accuracy_info}"
                    )
                    print(status_line)
                
                # Simulate packet rate (wait between packets)
                if packet_rate > 0:
                    time_per_packet = 1.0 / packet_rate
                    processing_time = time.time() - packet_arrival
                    sleep_time = max(0, time_per_packet - processing_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
        
        total_time = time.time() - self.start_time
        
        # Final summary
        print("\n" + "="*80)
        print(" SIMULATION SUMMARY")
        print("="*80)
        print(f"Simulation Duration: {total_time:.2f} seconds")
        print(f"Total Packets Processed: {self.total_packets}")
        print(f"Normal Traffic: {self.normal_packets} ({self.normal_packets/self.total_packets*100:.2f}%)")
        print(f"Attack Traffic: {self.attack_packets} ({self.attack_packets/self.total_packets*100:.2f}%)")
        print(f"Total Alerts Generated: {len(self.attack_alerts)}")
        print(f"Average Processing Time: {np.mean(self.processing_times)*1000:.2f}ms per packet")
        print(f"Actual Processing Rate: {self.total_packets/total_time:.2f} packets/second")
        print("="*80)
        
        # Performance metrics if labels available
        if y_true is not None:
            all_probs = np.array(all_probs)
            all_preds = np.array(all_preds)
            y_subset = y_true[indices[:len(all_probs)]]
            metrics = evaluate_binary(y_subset, all_probs)
            
            # Calculate confusion matrix details
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_subset, all_preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            print("\n DETECTION PERFORMANCE & VALIDATION")
            print("="*80)
            print("Classification Metrics:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
            print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
            print(f"  F1 Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
            print("\nConfusion Matrix:")
            print(f"  True Negatives (Normal correctly identified):  {tn}")
            print(f"  False Positives (Normal misclassified as Attack): {fp}")
            print(f"  False Negatives (Attack misclassified as Normal): {fn}")
            print(f"  True Positives (Attack correctly identified):   {tp}")
            print(f"\n  Total Normal Packets: {tn + fp}")
            print(f"  Total Attack Packets:  {fn + tp}")
            print("="*80)
        
        return {
            'probabilities': np.array(all_probs),
            'predictions': np.array(all_preds),
            'attack_alerts': self.attack_alerts,
            'statistics': {
                'total_packets': self.total_packets,
                'normal_packets': self.normal_packets,
                'attack_packets': self.attack_packets,
                'total_time': total_time,
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0
            }
        }
    
    def save_simulation_report(self, results, y_true=None, output_dir="simulator_results"):
        """Save detailed simulation report with validation metrics."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save attack alerts
        if results['attack_alerts']:
            alerts_df = pd.DataFrame(results['attack_alerts'])
            alerts_path = os.path.join(output_dir, "attack_alerts.csv")
            alerts_df.to_csv(alerts_path, index=False)
            print(f"\n Attack alerts saved to: {alerts_path}")
        
        # Save all predictions with validation
        results_df = pd.DataFrame({
            'packet_id': range(len(results['probabilities'])),
            'probability': results['probabilities'],
            'prediction': results['predictions'],
            'prediction_label': ['ATTACK' if p == 1 else 'NORMAL' for p in results['predictions']]
        })
        
        if y_true is not None:
            y_subset = y_true[:len(results['predictions'])]
            results_df['true_label'] = y_subset
            results_df['true_label_name'] = ['ATTACK' if y == 1 else 'NORMAL' for y in y_subset]
            results_df['correct'] = (results_df['prediction'] == results_df['true_label'])
            results_df['error_type'] = results_df.apply(
                lambda row: 'FP' if (row['prediction'] == 1 and row['true_label'] == 0) 
                else ('FN' if (row['prediction'] == 0 and row['true_label'] == 1) else 'Correct'),
                axis=1
            )
        
        csv_path = os.path.join(output_dir, "simulation_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f" All predictions saved to: {csv_path}")
        
        # Calculate detailed metrics
        from sklearn.metrics import confusion_matrix, classification_report
        
        if y_true is not None:
            y_subset = y_true[:len(results['predictions'])]
            cm = confusion_matrix(y_subset, results['predictions'], labels=[0, 1])
            metrics = evaluate_binary(y_subset, results['probabilities'])
            
            # Save detailed validation report
            validation_path = os.path.join(output_dir, "validation_report.txt")
            with open(validation_path, "w") as f:
                f.write("NIDS Simulation Validation Report\n")
                f.write("="*70 + "\n\n")
                f.write(f"Model: FT-Transformer\n")
                f.write(f"Model Path: {self.model_path}\n")
                f.write(f"Threshold: {self.threshold}\n")
                f.write(f"Simulation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("PERFORMANCE METRICS\n")
                f.write("-"*70 + "\n")
                f.write(f"Accuracy:  {metrics['accuracy']:.6f} ({metrics['accuracy']*100:.2f}%)\n")
                f.write(f"Precision: {metrics['precision']:.6f} ({metrics['precision']*100:.2f}%)\n")
                f.write(f"Recall:    {metrics['recall']:.6f} ({metrics['recall']*100:.2f}%)\n")
                f.write(f"F1 Score:  {metrics['f1']:.6f} ({metrics['f1']*100:.2f}%)\n\n")
                
                f.write("CONFUSION MATRIX\n")
                f.write("-"*70 + "\n")
                f.write("                Predicted\n")
                f.write("              Normal  Attack\n")
                f.write(f"Actual Normal   {cm[0,0]:5d}   {cm[0,1]:5d}\n")
                f.write(f"        Attack   {cm[1,0]:5d}   {cm[1,1]:5d}\n\n")
                
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                f.write("DETAILED BREAKDOWN\n")
                f.write("-"*70 + "\n")
                f.write(f"True Negatives (TN):  {tn:6d} - Normal correctly identified as Normal\n")
                f.write(f"False Positives (FP): {fp:6d} - Normal incorrectly flagged as Attack\n")
                f.write(f"False Negatives (FN): {fn:6d} - Attack missed (classified as Normal)\n")
                f.write(f"True Positives (TP):  {tp:6d} - Attack correctly detected\n\n")
                
                f.write(f"Total Normal Packets: {tn + fp}\n")
                f.write(f"Total Attack Packets: {fn + tp}\n\n")
                
                # Error analysis
                if fp > 0 or fn > 0:
                    f.write("ERROR ANALYSIS\n")
                    f.write("-"*70 + "\n")
                    f.write(f"False Positive Rate: {fp/(tn+fp)*100:.2f}% ({fp} out of {tn+fp} normal packets)\n")
                    f.write(f"False Negative Rate: {fn/(fn+tp)*100:.2f}% ({fn} out of {fn+tp} attack packets)\n")
            
            print(f" Validation report saved to: {validation_path}")
        
        # Save summary report
        summary_path = os.path.join(output_dir, "simulation_report.txt")
        with open(summary_path, "w") as f:
            f.write("NIDS Simulation Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Model: FT-Transformer\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Threshold: {self.threshold}\n")
            f.write(f"Simulation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            stats = results['statistics']
            f.write("Statistics:\n")
            f.write(f"  Total Packets: {stats['total_packets']}\n")
            f.write(f"  Normal Traffic: {stats['normal_packets']} ({stats['normal_packets']/stats['total_packets']*100:.2f}%)\n")
            f.write(f"  Attack Traffic: {stats['attack_packets']} ({stats['attack_packets']/stats['total_packets']*100:.2f}%)\n")
            f.write(f"  Total Alerts: {len(results['attack_alerts'])}\n")
            f.write(f"  Simulation Duration: {stats['total_time']:.2f} seconds\n")
            f.write(f"  Avg Processing Time: {stats['avg_processing_time']*1000:.2f}ms per packet\n")
            f.write(f"  Processing Rate: {stats['total_packets']/stats['total_time']:.2f} packets/second\n")
        
        print(f" Simulation report saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="NIDS Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulate 60 seconds of network monitoring at 10 packets/second
  python simulator.py --model_path results/ftt/ftt_best.pt --data_dir Datasets/UNSW --duration 60 --packet_rate 10
  
  # Simulate high-speed monitoring (100 packets/second)
  python simulator.py --model_path results/ftt/ftt_best.pt --data_dir Datasets/UNSW --packet_rate 100
  
  # Process all test data in simulation mode
  python simulator.py --model_path results/ftt/ftt_best.pt --data_dir Datasets/UNSW --packet_rate 100
        """
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to FT-Transformer checkpoint (.pt file)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory containing UNSW_NB15 CSV files")
    parser.add_argument("--packet_rate", type=int, default=10,
                        help="Simulated packet rate (packets per second)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Simulation duration in seconds (None = process all available data)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for attack detection (0-1)")
    parser.add_argument("--alert_cooldown", type=int, default=5,
                        help="Minimum seconds between alert displays")
    parser.add_argument("--no_alerts", action="store_true",
                        help="Disable real-time attack alerts")
    parser.add_argument("--output_dir", type=str, default="simulator_results",
                        help="Directory to save simulation results")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NIDS] Using device: {device}")
    
    # Load data
    print(f"[NIDS] Loading network traffic data from: {args.data_dir}")
    (X_train_full, X_test_full,
     X_train_num, X_test_num,
     X_train_cat, X_test_cat,
     y_train, y_test,
     num_cols, cat_cols,
     scaler) = load_unsw_kaggle_dataset(args.data_dir)
    
    # Use test data for simulation
    X_num = X_test_num
    X_cat = X_test_cat
    y_true = y_test
    
    # Calculate FTT categories from train+test
    all_cat = np.vstack([X_train_cat, X_test_cat]) if X_test_cat.shape[0] > 0 else X_train_cat
    categories = tuple(int(np.max(all_cat[:, i]) + 1) for i in range(X_train_cat.shape[1]))
    num_cont = X_train_num.shape[1]
    
    # Calculate pos_weight_ratio
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight_ratio = max(1.0, n_neg / max(1, n_pos))
    
    # Initialize simulator
    simulator = NIDSSimulator(
        model_path=args.model_path,
        device=device,
        batch_size=1,  # Process one at a time for real-time simulation
        threshold=args.threshold
    )
    
    # Load model
    simulator.load_model(categories, num_cont, pos_weight_ratio)
    
    # Run simulation
    results = simulator.simulate_network_traffic(
        X_num, X_cat, 
        y_true=y_true,
        packet_rate=args.packet_rate,
        duration=args.duration,
        show_alerts=not args.no_alerts,
        alert_cooldown=args.alert_cooldown
    )
    
    # Save results
    simulator.save_simulation_report(results, y_true=y_true, output_dir=args.output_dir)
    
    print("\n Simulation completed!")


if __name__ == "__main__":
    main()
