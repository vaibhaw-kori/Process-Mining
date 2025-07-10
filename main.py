import pm4py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class ConceptDriftAnalyzer:
    def __init__(self, log_path):
        """Initialize the analyzer with the event log"""
        self.log_path = log_path
        self.event_log = None
        self.dataframe = None
        self.drift_results = {}
        self.report_sections = []
        
    def load_data(self):
        """Load the event log from XES file"""
        try:
            self.event_log = pm4py.read_xes(self.log_path)
            self.dataframe = pm4py.convert_to_dataframe(self.event_log)
            print(f"✓ Successfully loaded {len(self.event_log)} traces with {len(self.dataframe)} events")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return False
    
    def basic_statistics(self):
        """Generate basic statistics about the event log"""
        stats = {
            'total_traces': len(self.event_log),
            'total_events': len(self.dataframe),
            'unique_activities': self.dataframe['concept:name'].nunique(),
            'unique_resources': self.dataframe['org:resource'].nunique() if 'org:resource' in self.dataframe.columns else 0,
            'time_span': None,
            'activity_distribution': dict(self.dataframe['concept:name'].value_counts()),
            'trace_length_stats': {}
        }
        
        # Time span analysis
        if 'time:timestamp' in self.dataframe.columns:
            self.dataframe['time:timestamp'] = pd.to_datetime(self.dataframe['time:timestamp'])
            stats['time_span'] = {
                'start': self.dataframe['time:timestamp'].min(),
                'end': self.dataframe['time:timestamp'].max(),
                'duration_days': (self.dataframe['time:timestamp'].max() - 
                                self.dataframe['time:timestamp'].min()).days
            }
        
        # Trace length statistics - using dataframe approach
        try:
            trace_lengths = self.dataframe.groupby('case:concept:name').size().values
            stats['trace_length_stats'] = {
                'mean': np.mean(trace_lengths),
                'median': np.median(trace_lengths),
                'std': np.std(trace_lengths),
                'min': int(np.min(trace_lengths)),
                'max': int(np.max(trace_lengths))
            }
        except Exception as e:
            print(f"Warning: Could not calculate trace length statistics: {e}")
            stats['trace_length_stats'] = {
                'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0
            }
        
        self.basic_stats = stats
        return stats
    
    def detect_temporal_drift(self, time_windows=10):
        """Detect concept drift over time using multiple approaches"""
        if 'time:timestamp' not in self.dataframe.columns:
            return {"error": "No timestamp information available"}
        
        # Sort by timestamp
        df_sorted = self.dataframe.sort_values('time:timestamp')
        
        # Create time windows
        total_events = len(df_sorted)
        window_size = total_events // time_windows
        
        temporal_analysis = {
            'activity_drift': {},
            'resource_drift': {},
            'throughput_drift': {},
            'trace_variant_drift': {},
            'window_stats': []
        }
        
        for i in range(time_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, total_events)
            window_data = df_sorted.iloc[start_idx:end_idx]
            
            window_stats = {
                'window': i + 1,
                'start_time': window_data['time:timestamp'].min(),
                'end_time': window_data['time:timestamp'].max(),
                'event_count': len(window_data),
                'unique_traces': window_data['case:concept:name'].nunique(),
                'activity_distribution': dict(window_data['concept:name'].value_counts(normalize=True)),
                'resource_distribution': dict(window_data['org:resource'].value_counts(normalize=True)) if 'org:resource' in window_data.columns else {}
            }
            
            temporal_analysis['window_stats'].append(window_stats)
        
        # Calculate drift metrics
        self._calculate_drift_metrics(temporal_analysis)
        
        return temporal_analysis
    
    def _calculate_drift_metrics(self, temporal_analysis):
        """Calculate various drift metrics between time windows"""
        windows = temporal_analysis['window_stats']
        
        # Activity distribution drift (using Jensen-Shannon divergence)
        activity_drift_scores = []
        resource_drift_scores = []
        
        for i in range(1, len(windows)):
            prev_activities = windows[i-1]['activity_distribution']
            curr_activities = windows[i]['activity_distribution']
            
            # Calculate JS divergence for activities
            js_div = self._jensen_shannon_divergence(prev_activities, curr_activities)
            activity_drift_scores.append({
                'window_pair': f"{i}-{i+1}",
                'js_divergence': js_div,
                'drift_detected': js_div > 0.1  # Threshold for drift detection
            })
            
            # Resource drift if available
            if windows[i-1]['resource_distribution'] and windows[i]['resource_distribution']:
                prev_resources = windows[i-1]['resource_distribution']
                curr_resources = windows[i]['resource_distribution']
                js_div_res = self._jensen_shannon_divergence(prev_resources, curr_resources)
                resource_drift_scores.append({
                    'window_pair': f"{i}-{i+1}",
                    'js_divergence': js_div_res,
                    'drift_detected': js_div_res > 0.1
                })
        
        temporal_analysis['activity_drift'] = activity_drift_scores
        temporal_analysis['resource_drift'] = resource_drift_scores
    
    def _jensen_shannon_divergence(self, p, q):
        """Calculate Jensen-Shannon divergence between two probability distributions"""
        # Get all unique keys
        all_keys = set(list(p.keys()) + list(q.keys()))
        
        # Convert to arrays with same keys
        p_array = np.array([p.get(key, 1e-10) for key in all_keys])
        q_array = np.array([q.get(key, 1e-10) for key in all_keys])
        
        # Normalize
        p_array = p_array / np.sum(p_array)
        q_array = q_array / np.sum(q_array)
        
        # Calculate JS divergence
        m = 0.5 * (p_array + q_array)
        js_div = 0.5 * self._kl_divergence(p_array, m) + 0.5 * self._kl_divergence(q_array, m)
        
        return js_div
    
    def _kl_divergence(self, p, q):
        """Calculate KL divergence"""
        return np.sum(p * np.log(p / q + 1e-10))
    
    def detect_process_variant_drift(self):
        """Detect drift in process variants (directly-follows relationships)"""
        try:
            # Get directly-follows graphs for different time periods
            df_sorted = self.dataframe.sort_values('time:timestamp')
            
            # Split into two halves
            mid_point = len(df_sorted) // 2
            first_half = df_sorted.iloc[:mid_point]
            second_half = df_sorted.iloc[mid_point:]
            
            # Convert back to event logs safely
            try:
                first_log = pm4py.convert_to_event_log(first_half)
                second_log = pm4py.convert_to_event_log(second_half)
            except Exception as e:
                print(f"Warning: Could not convert to event logs: {e}")
                return {"error": "Could not convert dataframes to event logs"}
            
            # Get directly-follows graphs
            dfg1, start1, end1 = pm4py.discover_dfg(first_log)
            dfg2, start2, end2 = pm4py.discover_dfg(second_log)
            
            # Compare DFGs
            variant_drift_analysis = {
                'first_half_relations': len(dfg1),
                'second_half_relations': len(dfg2),
                'common_relations': len(set(dfg1.keys()) & set(dfg2.keys())),
                'new_relations': list(set(dfg2.keys()) - set(dfg1.keys())),
                'disappeared_relations': list(set(dfg1.keys()) - set(dfg2.keys())),
                'relation_frequency_changes': []
            }
            
            # Analyze frequency changes in common relations
            common_relations = set(dfg1.keys()) & set(dfg2.keys())
            for relation in common_relations:
                freq1 = dfg1[relation]
                freq2 = dfg2[relation]
                if freq1 > 0:  # Avoid division by zero
                    change_ratio = freq2 / freq1
                    if abs(change_ratio - 1) > 0.5:  # Significant change threshold
                        variant_drift_analysis['relation_frequency_changes'].append({
                            'relation': relation,
                            'first_half_freq': freq1,
                            'second_half_freq': freq2,
                            'change_ratio': change_ratio
                        })
            
            return variant_drift_analysis
            
        except Exception as e:
            print(f"Warning: Process variant drift analysis failed: {e}")
            return {
                'error': f"Process variant analysis failed: {str(e)}",
                'first_half_relations': 0,
                'second_half_relations': 0,
                'common_relations': 0,
                'new_relations': [],
                'disappeared_relations': [],
                'relation_frequency_changes': []
            }
    
    def detect_performance_drift(self):
        """Detect drift in performance metrics"""
        if 'time:timestamp' not in self.dataframe.columns:
            return {"error": "No timestamp information available"}
        
        # Calculate case durations using dataframe approach
        case_durations = []
        
        # Group by case and calculate durations
        for case_id, case_events in self.dataframe.groupby('case:concept:name'):
            case_events_sorted = case_events.sort_values('time:timestamp')
            
            if len(case_events_sorted) > 0:
                start_time = case_events_sorted['time:timestamp'].iloc[0]
                end_time = case_events_sorted['time:timestamp'].iloc[-1]
                
                # Handle duration calculation safely
                try:
                    if pd.notna(start_time) and pd.notna(end_time):
                        duration = (end_time - start_time).total_seconds() / 3600  # in hours
                        case_durations.append({
                            'case_id': case_id,
                            'duration_hours': max(0, duration),  # Ensure non-negative
                            'start_time': start_time,
                            'end_time': end_time,
                            'num_events': len(case_events_sorted)
                        })
                except Exception as e:
                    print(f"Warning: Could not calculate duration for case {case_id}: {e}")
                    continue
        
        if not case_durations:
            return {"error": "No valid case durations could be calculated"}
        
        df_durations = pd.DataFrame(case_durations)
        df_durations = df_durations.sort_values('start_time')
        
        # Split into time windows and analyze performance drift
        n_windows = 5
        window_size = len(df_durations) // n_windows
        
        performance_analysis = {
            'duration_drift': [],
            'throughput_drift': [],
            'complexity_drift': []
        }
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(df_durations))
            window_data = df_durations.iloc[start_idx:end_idx]
            
            performance_analysis['duration_drift'].append({
                'window': i + 1,
                'mean_duration': window_data['duration_hours'].mean(),
                'median_duration': window_data['duration_hours'].median(),
                'std_duration': window_data['duration_hours'].std()
            })
            
            performance_analysis['complexity_drift'].append({
                'window': i + 1,
                'mean_events_per_case': window_data['num_events'].mean(),
                'median_events_per_case': window_data['num_events'].median()
            })
        
        return performance_analysis
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive drift analysis report"""
        print("🔍 Starting comprehensive concept drift analysis...")
        
        # Load data
        if not self.load_data():
            return "Failed to load data"
        
        print(f"📊 Analyzing {len(self.dataframe):,} events across {len(self.event_log):,} traces...")
        
        # Generate all analyses with progress indicators
        print("1/4 Calculating basic statistics...")
        basic_stats = self.basic_statistics()
        
        print("2/4 Analyzing temporal drift patterns...")
        temporal_drift = self.detect_temporal_drift()
        
        print("3/4 Detecting process variant drift...")
        variant_drift = self.detect_process_variant_drift()
        
        print("4/4 Analyzing performance drift...")
        performance_drift = self.detect_performance_drift()
        
        print("✅ Analysis complete! Generating report...")
        
        # Generate report
        report = self._format_comprehensive_report(
            basic_stats, temporal_drift, variant_drift, performance_drift
        )
        
        return report
    
    def _format_comprehensive_report(self, basic_stats, temporal_drift, variant_drift, performance_drift):
        """Format the comprehensive analysis report"""
        
        report = f"""
# COMPREHENSIVE CONCEPT DRIFT ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
This report analyzes concept drift in the provided business process event log, focusing on temporal changes in activity patterns, process variants, resource utilization, and performance metrics.

## 1. DATASET OVERVIEW

### Basic Statistics:
- **Total Process Instances (Traces):** {basic_stats['total_traces']:,}
- **Total Events:** {basic_stats['total_events']:,}
- **Unique Activities:** {basic_stats['unique_activities']}
- **Unique Resources:** {basic_stats['unique_resources']}
- **Time Span:** {basic_stats['time_span']['duration_days'] if basic_stats['time_span'] else 'N/A'} days
- **Average Trace Length:** {basic_stats['trace_length_stats']['mean']:.2f} events
- **Trace Length Range:** {basic_stats['trace_length_stats']['min']} - {basic_stats['trace_length_stats']['max']} events

### Activity Distribution:
"""
        
        # Add activity distribution
        for activity, count in list(basic_stats['activity_distribution'].items())[:10]:
            percentage = (count / basic_stats['total_events']) * 100
            report += f"- **{activity}:** {count:,} events ({percentage:.1f}%)\n"
        
        report += f"""

## 2. TEMPORAL DRIFT ANALYSIS

### Activity Pattern Drift:
"""
        
        # Analyze activity drift
        drift_detected = False
        high_drift_windows = []
        
        if 'activity_drift' in temporal_drift:
            for drift_score in temporal_drift['activity_drift']:
                if drift_score['drift_detected']:
                    drift_detected = True
                    high_drift_windows.append(drift_score)
            
            if drift_detected:
                report += f"⚠️ **SIGNIFICANT DRIFT DETECTED** in {len(high_drift_windows)} window transitions:\n"
                for drift in high_drift_windows[:5]:  # Show top 5
                    report += f"- Windows {drift['window_pair']}: JS Divergence = {drift['js_divergence']:.3f}\n"
            else:
                report += "✅ **NO SIGNIFICANT ACTIVITY DRIFT** detected across time windows.\n"
        
        report += f"""

### Resource Utilization Drift:
"""
        
        # Analyze resource drift
        resource_drift_detected = False
        if 'resource_drift' in temporal_drift and temporal_drift['resource_drift']:
            high_resource_drift = [d for d in temporal_drift['resource_drift'] if d['drift_detected']]
            if high_resource_drift:
                resource_drift_detected = True
                report += f"⚠️ **RESOURCE DRIFT DETECTED** in {len(high_resource_drift)} window transitions.\n"
            else:
                report += "✅ **NO SIGNIFICANT RESOURCE DRIFT** detected.\n"
        else:
            report += "ℹ️ Resource drift analysis not available (no resource information).\n"
        
        report += f"""

## 3. PROCESS VARIANT DRIFT ANALYSIS

### Directly-Follows Relationship Changes:
- **First Half Relations:** {variant_drift['first_half_relations']}
- **Second Half Relations:** {variant_drift['second_half_relations']}
- **Common Relations:** {variant_drift['common_relations']}
- **New Relations:** {len(variant_drift['new_relations'])}
- **Disappeared Relations:** {len(variant_drift['disappeared_relations'])}

"""
        
        if variant_drift['new_relations']:
            report += "### New Process Paths Detected:\n"
            for relation in variant_drift['new_relations'][:10]:
                report += f"- {relation[0]} → {relation[1]}\n"
        
        if variant_drift['disappeared_relations']:
            report += "\n### Disappeared Process Paths:\n"
            for relation in variant_drift['disappeared_relations'][:10]:
                report += f"- {relation[0]} → {relation[1]}\n"
        
        if variant_drift['relation_frequency_changes']:
            report += f"\n### Significant Frequency Changes ({len(variant_drift['relation_frequency_changes'])} detected):\n"
            for change in variant_drift['relation_frequency_changes'][:5]:
                report += f"- **{change['relation'][0]} → {change['relation'][1]}:** {change['change_ratio']:.2f}x change\n"
        
        report += f"""

## 4. PERFORMANCE DRIFT ANALYSIS

### Case Duration Trends:
"""
        
        if 'error' not in performance_drift:
            duration_windows = performance_drift['duration_drift']
            first_window = duration_windows[0]
            last_window = duration_windows[-1]
            
            duration_change = ((last_window['mean_duration'] - first_window['mean_duration']) / 
                             first_window['mean_duration']) * 100
            
            report += f"- **Initial Average Duration:** {first_window['mean_duration']:.2f} hours\n"
            report += f"- **Final Average Duration:** {last_window['mean_duration']:.2f} hours\n"
            report += f"- **Overall Duration Change:** {duration_change:+.1f}%\n"
            
            if abs(duration_change) > 20:
                report += f"⚠️ **SIGNIFICANT PERFORMANCE DRIFT** detected ({duration_change:+.1f}% change)\n"
            else:
                report += "✅ **STABLE PERFORMANCE** maintained over time.\n"
        else:
            report += "ℹ️ Performance analysis not available (no timestamp information).\n"
        
        report += f"""

## 5. DRIFT DETECTION SUMMARY

### Overall Assessment:
"""
        
        total_drift_indicators = 0
        drift_types = []
        
        if drift_detected:
            total_drift_indicators += 1
            drift_types.append("Activity Pattern Drift")
        
        if resource_drift_detected:
            total_drift_indicators += 1
            drift_types.append("Resource Utilization Drift")
        
        if len(variant_drift['new_relations']) > 5 or len(variant_drift['disappeared_relations']) > 5:
            total_drift_indicators += 1
            drift_types.append("Process Variant Drift")
        
        if 'error' not in performance_drift:
            duration_windows = performance_drift['duration_drift']
            if len(duration_windows) >= 2:
                first_duration = duration_windows[0]['mean_duration']
                last_duration = duration_windows[-1]['mean_duration']
                if abs((last_duration - first_duration) / first_duration) > 0.2:
                    total_drift_indicators += 1
                    drift_types.append("Performance Drift")
        
        if total_drift_indicators == 0:
            report += "✅ **MINIMAL CONCEPT DRIFT DETECTED** - Process appears stable over time.\n"
        elif total_drift_indicators <= 2:
            report += f"⚠️ **MODERATE CONCEPT DRIFT DETECTED** - {total_drift_indicators} drift types identified:\n"
        else:
            report += f"🚨 **SIGNIFICANT CONCEPT DRIFT DETECTED** - {total_drift_indicators} drift types identified:\n"
        
        for drift_type in drift_types:
            report += f"- {drift_type}\n"
        
        report += f"""

## 6. RECOMMENDATIONS

### Process Monitoring:
- Monitor the identified drift patterns continuously
- Set up alerts for significant deviations in key metrics
- Review process documentation and training materials

### Process Improvement:
"""
        
        if drift_detected:
            report += "- Investigate root causes of activity pattern changes\n"
        
        if resource_drift_detected:
            report += "- Review resource allocation and workload distribution\n"
        
        if len(variant_drift['new_relations']) > 0:
            report += "- Analyze new process paths for compliance and efficiency\n"
        
        if total_drift_indicators > 2:
            report += "- Consider process redesign to address multiple drift sources\n"
        
        report += """
- Implement regular process mining analysis cycles
- Establish baseline metrics for future drift detection

---
*This report was generated using PM4py and automated concept drift detection algorithms.*
*For detailed technical analysis, please refer to the underlying drift detection metrics.*
"""
        
        return report

# Usage Example and Main Execution
def main():
    # Initialize analyzer with your XES file path
    analyzer = ConceptDriftAnalyzer('E:\process_mining\Concept Drift PM\BPI Challenge 2017.xes')  # Update with your file path
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    # Save report to file
    with open('concept_drift_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("📊 Comprehensive drift analysis complete!")
    print("📄 Report saved to: concept_drift_analysis_report.txt")
    print("\n" + "="*80)
    print(report)
    
    return analyzer, report

if __name__ == "__main__":
    analyzer, report = main()