//! Benchmark report generation.

use serde::{Deserialize, Serialize};
use std::io::Write;
use tabled::{Table, Tabled};

use crate::{
    TransferResult, TransferSummary,
    DecisionResult, DecisionSummary,
    MemoryResult, MemorySummary,
};

/// Report output format.
#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    /// Plain text tables.
    Text,
    /// JSON format.
    Json,
    /// Markdown format.
    Markdown,
    /// CSV format.
    Csv,
}

/// Complete benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Report title.
    pub title: String,
    /// Timestamp.
    pub timestamp: String,
    /// System information.
    pub system_info: SystemInfo,
    /// Transfer benchmark results.
    pub transfer_results: Option<Vec<TransferResult>>,
    /// Transfer summary.
    pub transfer_summary: Option<TransferSummary>,
    /// Decision benchmark results.
    pub decision_results: Option<Vec<DecisionResult>>,
    /// Decision summary.
    pub decision_summary: Option<DecisionSummary>,
    /// Memory benchmark results.
    pub memory_results: Option<Vec<MemoryResult>>,
    /// Memory summary.
    pub memory_summary: Option<MemorySummary>,
}

/// System information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub gpu_name: String,
    pub gpu_memory_gb: f64,
    pub cuda_version: String,
    pub driver_version: String,
    pub cpu_info: String,
    pub os_info: String,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            gpu_name: "Unknown GPU".to_string(),
            gpu_memory_gb: 0.0,
            cuda_version: "Unknown".to_string(),
            driver_version: "Unknown".to_string(),
            cpu_info: "Unknown".to_string(),
            os_info: std::env::consts::OS.to_string(),
        }
    }
}

impl BenchmarkReport {
    /// Create a new report.
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            timestamp: chrono_lite(),
            system_info: SystemInfo::default(),
            transfer_results: None,
            transfer_summary: None,
            decision_results: None,
            decision_summary: None,
            memory_results: None,
            memory_summary: None,
        }
    }

    /// Add transfer results.
    pub fn with_transfer_results(mut self, results: Vec<TransferResult>) -> Self {
        self.transfer_summary = Some(TransferSummary::from_results(&results));
        self.transfer_results = Some(results);
        self
    }

    /// Add decision results.
    pub fn with_decision_results(mut self, results: Vec<DecisionResult>) -> Self {
        self.decision_summary = Some(DecisionSummary::from_results(&results));
        self.decision_results = Some(results);
        self
    }

    /// Add memory results.
    pub fn with_memory_results(mut self, results: Vec<MemoryResult>) -> Self {
        self.memory_summary = Some(MemorySummary::from_results(&results));
        self.memory_results = Some(results);
        self
    }

    /// Format report as string.
    pub fn format(&self, format: ReportFormat) -> String {
        match format {
            ReportFormat::Text => self.format_text(),
            ReportFormat::Json => self.format_json(),
            ReportFormat::Markdown => self.format_markdown(),
            ReportFormat::Csv => self.format_csv(),
        }
    }

    /// Format as plain text.
    fn format_text(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!("╔{'═'.repeat(70)}╗\n"));
        out.push_str(&format!("║ {:^68} ║\n", self.title));
        out.push_str(&format!("║ {:^68} ║\n", self.timestamp));
        out.push_str(&format!("╚{'═'.repeat(70)}╝\n\n"));

        // System info
        out.push_str("System Information:\n");
        out.push_str(&format!("  GPU: {}\n", self.system_info.gpu_name));
        out.push_str(&format!("  GPU Memory: {:.1} GB\n", self.system_info.gpu_memory_gb));
        out.push_str(&format!("  CUDA: {}\n", self.system_info.cuda_version));
        out.push_str(&format!("  OS: {}\n\n", self.system_info.os_info));

        // Transfer results
        if let Some(ref results) = self.transfer_results {
            out.push_str("Transfer Benchmarks:\n");
            out.push_str(&format!("{:-<70}\n", ""));

            let headers = ["Direction", "Size", "Time (ms)", "BW (GB/s)", "StdDev"];
            out.push_str(&format!(
                "{:>15} {:>12} {:>12} {:>12} {:>12}\n",
                headers[0], headers[1], headers[2], headers[3], headers[4]
            ));
            out.push_str(&format!("{:-<70}\n", ""));

            for result in results {
                let row = result.to_row();
                out.push_str(&format!(
                    "{:>15} {:>12} {:>12} {:>12} {:>12}\n",
                    row[0], row[1], row[2], row[3], row[4]
                ));
            }
            out.push('\n');

            if let Some(ref summary) = self.transfer_summary {
                out.push_str("Transfer Summary:\n");
                out.push_str(&format!("  Peak H2D Bandwidth: {:.2} GB/s\n", summary.peak_h2d_bandwidth));
                out.push_str(&format!("  Peak D2H Bandwidth: {:.2} GB/s\n", summary.peak_d2h_bandwidth));
                out.push_str(&format!("  Peak D2D Bandwidth: {:.2} GB/s\n", summary.peak_d2d_bandwidth));
                out.push('\n');
            }
        }

        // Decision results
        if let Some(ref results) = self.decision_results {
            out.push_str("Decision Engine Benchmarks:\n");
            out.push_str(&format!("{:-<70}\n", ""));

            let headers = ["States", "Ops", "Pressure", "Time (ms)", "Decisions", "Dec/ms"];
            out.push_str(&format!(
                "{:>10} {:>8} {:>10} {:>12} {:>12} {:>10}\n",
                headers[0], headers[1], headers[2], headers[3], headers[4], headers[5]
            ));
            out.push_str(&format!("{:-<70}\n", ""));

            for result in results {
                let row = result.to_row();
                out.push_str(&format!(
                    "{:>10} {:>8} {:>10} {:>12} {:>12} {:>10}\n",
                    row[0], row[1], row[2], row[3], row[4], row[5]
                ));
            }
            out.push('\n');

            if let Some(ref summary) = self.decision_summary {
                out.push_str("Decision Summary:\n");
                out.push_str(&format!("  Avg Planning Time: {:.3} ms\n", summary.avg_planning_time_ms));
                out.push_str(&format!("  Max Planning Time: {:.3} ms\n", summary.max_planning_time_ms));
                out.push_str(&format!("  Avg Decisions/ms: {:.0}\n", summary.avg_decisions_per_ms));
                out.push('\n');
            }
        }

        // Memory results
        if let Some(ref results) = self.memory_results {
            out.push_str("Memory Management Benchmarks:\n");
            out.push_str(&format!("{:-<70}\n", ""));

            let headers = ["Test", "Allocs", "Total", "Alloc (ms)", "Free (ms)", "Allocs/s"];
            out.push_str(&format!(
                "{:<25} {:>8} {:>10} {:>12} {:>10} {:>10}\n",
                headers[0], headers[1], headers[2], headers[3], headers[4], headers[5]
            ));
            out.push_str(&format!("{:-<70}\n", ""));

            for result in results {
                let row = result.to_row();
                out.push_str(&format!(
                    "{:<25} {:>8} {:>10} {:>12} {:>10} {:>10}\n",
                    row[0], row[1], row[2], row[3], row[4], row[5]
                ));
            }
            out.push('\n');

            if let Some(ref summary) = self.memory_summary {
                out.push_str("Memory Summary:\n");
                out.push_str(&format!("  Pool Speedup: {:.1}x\n", summary.pool_speedup));
                out.push_str(&format!("  Avg Allocs/sec: {:.0}\n", summary.avg_allocs_per_second));
                out.push('\n');
            }
        }

        out
    }

    /// Format as JSON.
    fn format_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Format as Markdown.
    fn format_markdown(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!("# {}\n\n", self.title));
        out.push_str(&format!("*Generated: {}*\n\n", self.timestamp));

        out.push_str("## System Information\n\n");
        out.push_str(&format!("- **GPU**: {}\n", self.system_info.gpu_name));
        out.push_str(&format!("- **GPU Memory**: {:.1} GB\n", self.system_info.gpu_memory_gb));
        out.push_str(&format!("- **CUDA**: {}\n", self.system_info.cuda_version));
        out.push_str(&format!("- **OS**: {}\n\n", self.system_info.os_info));

        // Transfer results
        if let Some(ref results) = self.transfer_results {
            out.push_str("## Transfer Benchmarks\n\n");
            out.push_str("| Direction | Size | Time (ms) | Bandwidth (GB/s) | StdDev |\n");
            out.push_str("|-----------|------|-----------|------------------|--------|\n");

            for result in results {
                let row = result.to_row();
                out.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    row[0], row[1], row[2], row[3], row[4]
                ));
            }
            out.push('\n');
        }

        // Decision results
        if let Some(ref results) = self.decision_results {
            out.push_str("## Decision Engine Benchmarks\n\n");
            out.push_str("| States | Ops | Pressure | Time (ms) | Decisions | Dec/ms |\n");
            out.push_str("|--------|-----|----------|-----------|-----------|--------|\n");

            for result in results {
                let row = result.to_row();
                out.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} |\n",
                    row[0], row[1], row[2], row[3], row[4], row[5]
                ));
            }
            out.push('\n');
        }

        // Memory results
        if let Some(ref results) = self.memory_results {
            out.push_str("## Memory Management Benchmarks\n\n");
            out.push_str("| Test | Allocs | Total | Alloc (ms) | Free (ms) | Allocs/s |\n");
            out.push_str("|------|--------|-------|------------|-----------|----------|\n");

            for result in results {
                let row = result.to_row();
                out.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} |\n",
                    row[0], row[1], row[2], row[3], row[4], row[5]
                ));
            }
            out.push('\n');
        }

        out
    }

    /// Format as CSV.
    fn format_csv(&self) -> String {
        let mut out = String::new();

        // Transfer results
        if let Some(ref results) = self.transfer_results {
            out.push_str("# Transfer Benchmarks\n");
            out.push_str("direction,size_bytes,time_ms,bandwidth_gbps,stddev_ms\n");

            for result in results {
                out.push_str(&format!(
                    "{},{},{},{},{}\n",
                    result.direction, result.size_bytes, result.time_ms,
                    result.bandwidth_gbps, result.time_stddev_ms
                ));
            }
            out.push('\n');
        }

        // Decision results
        if let Some(ref results) = self.decision_results {
            out.push_str("# Decision Benchmarks\n");
            out.push_str("num_states,num_operations,memory_pressure,planning_time_ms,num_decisions,decisions_per_ms\n");

            for result in results {
                out.push_str(&format!(
                    "{},{},{},{},{},{}\n",
                    result.num_states, result.num_operations, result.memory_pressure,
                    result.planning_time_ms, result.num_decisions, result.decisions_per_ms
                ));
            }
            out.push('\n');
        }

        // Memory results
        if let Some(ref results) = self.memory_results {
            out.push_str("# Memory Benchmarks\n");
            out.push_str("test_name,num_allocations,total_bytes,allocation_time_ms,deallocation_time_ms,allocs_per_second\n");

            for result in results {
                out.push_str(&format!(
                    "{},{},{},{},{},{}\n",
                    result.test_name, result.num_allocations, result.total_bytes,
                    result.allocation_time_ms, result.deallocation_time_ms,
                    result.allocs_per_second
                ));
            }
        }

        out
    }

    /// Write report to file.
    pub fn write_to_file(&self, path: &str, format: ReportFormat) -> std::io::Result<()> {
        let content = self.format(format);
        std::fs::write(path, content)
    }
}

/// Simple timestamp without chrono dependency.
fn chrono_lite() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = duration.as_secs();
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;
    let sec = secs % 60;

    // Approximate date (not accurate but sufficient for benchmarks)
    let year = 1970 + days / 365;
    let day_of_year = days % 365;

    format!(
        "{}-{:03} {:02}:{:02}:{:02} UTC",
        year, day_of_year, hours, mins, sec
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_creation() {
        let report = BenchmarkReport::new("Test Report");
        assert_eq!(report.title, "Test Report");
    }

    #[test]
    fn test_format_text() {
        let report = BenchmarkReport::new("Test Report");
        let text = report.format(ReportFormat::Text);
        assert!(text.contains("Test Report"));
    }

    #[test]
    fn test_format_json() {
        let report = BenchmarkReport::new("Test Report");
        let json = report.format(ReportFormat::Json);
        assert!(json.contains("\"title\""));
    }
}
