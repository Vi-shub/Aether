//! Aether Benchmark CLI
//!
//! Run comprehensive benchmarks on Aether runtime components.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use indicatif::{ProgressBar, ProgressStyle};

use aether_bench::{
    BenchmarkReport, ReportFormat,
    TransferBenchmark, DecisionBenchmark, MemoryBenchmark,
    scenarios::{TransferScenario, DecisionScenario, MemoryScenario, ModelConfig},
};

#[derive(Parser)]
#[command(name = "aether-bench")]
#[command(about = "Aether Runtime Benchmark Suite", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output format
    #[arg(short, long, default_value = "text")]
    format: OutputFormat,

    /// Output file (optional)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Verbosity level
    #[arg(short, long, default_value = "info")]
    verbosity: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Run all benchmarks
    All {
        /// Quick mode (fewer iterations)
        #[arg(short, long)]
        quick: bool,
    },
    /// Run transfer benchmarks
    Transfer {
        /// Number of iterations
        #[arg(short, long, default_value = "100")]
        iterations: usize,
        /// Warmup iterations
        #[arg(short, long, default_value = "10")]
        warmup: usize,
    },
    /// Run decision engine benchmarks
    Decision {
        /// Number of iterations per config
        #[arg(short, long, default_value = "100")]
        iterations: usize,
    },
    /// Run memory management benchmarks
    Memory {
        /// Pool size in GB
        #[arg(short, long, default_value = "8")]
        pool_size_gb: u64,
        /// Number of allocations
        #[arg(short = 'n', long, default_value = "1000")]
        num_allocations: usize,
    },
    /// Run inference scenario benchmark
    Inference {
        /// Model preset
        #[arg(short, long, default_value = "llama-7b")]
        model: ModelPreset,
        /// GPU memory limit in GB
        #[arg(short, long)]
        memory_limit: Option<u64>,
    },
    /// List available benchmarks
    List,
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Markdown,
    Csv,
}

impl From<OutputFormat> for ReportFormat {
    fn from(f: OutputFormat) -> Self {
        match f {
            OutputFormat::Text => ReportFormat::Text,
            OutputFormat::Json => ReportFormat::Json,
            OutputFormat::Markdown => ReportFormat::Markdown,
            OutputFormat::Csv => ReportFormat::Csv,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum ModelPreset {
    Llama7b,
    Llama70b,
    Mixtral8x7b,
    DeepseekV2,
}

impl ModelPreset {
    fn to_config(self) -> ModelConfig {
        match self {
            ModelPreset::Llama7b => ModelConfig::llama_7b(),
            ModelPreset::Llama70b => ModelConfig::llama_70b(),
            ModelPreset::Mixtral8x7b => ModelConfig::mixtral_8x7b(),
            ModelPreset::DeepseekV2 => ModelConfig::deepseek_v2(),
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let level = match cli.verbosity.as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Run benchmarks
    let report = match cli.command {
        Commands::All { quick } => run_all_benchmarks(quick)?,
        Commands::Transfer { iterations, warmup } => run_transfer_benchmarks(iterations, warmup)?,
        Commands::Decision { iterations } => run_decision_benchmarks(iterations)?,
        Commands::Memory { pool_size_gb, num_allocations } => {
            run_memory_benchmarks(pool_size_gb, num_allocations)?
        }
        Commands::Inference { model, memory_limit } => {
            run_inference_benchmark(model.to_config(), memory_limit)?
        }
        Commands::List => {
            print_available_benchmarks();
            return Ok(());
        }
    };

    // Output results
    let output = report.format(cli.format.into());

    if let Some(path) = cli.output {
        std::fs::write(&path, &output)?;
        info!("Report written to {}", path.display());
    } else {
        println!("{}", output);
    }

    Ok(())
}

fn run_all_benchmarks(quick: bool) -> anyhow::Result<BenchmarkReport> {
    let mut report = BenchmarkReport::new("Aether Complete Benchmark Suite");

    println!("\n🚀 Aether Benchmark Suite\n");

    let pb = ProgressBar::new(3);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({msg})")
        .unwrap()
        .progress_chars("#>-"));

    // Transfer benchmarks
    pb.set_message("Transfer benchmarks...");
    let transfer_scenario = if quick {
        TransferScenario {
            iterations: 10,
            warmup: 2,
            ..Default::default()
        }
    } else {
        TransferScenario::default()
    };

    if let Ok(bench) = TransferBenchmark::new(transfer_scenario) {
        let results = bench.run();
        report = report.with_transfer_results(results);
    }
    pb.inc(1);

    // Decision benchmarks
    pb.set_message("Decision benchmarks...");
    let decision_scenario = if quick {
        DecisionScenario {
            iterations: 10,
            num_states: vec![100, 1000],
            num_operations: vec![32],
            memory_pressures: vec![0.75],
        }
    } else {
        DecisionScenario::default()
    };

    let bench = DecisionBenchmark::new(decision_scenario);
    let results = bench.run();
    report = report.with_decision_results(results);
    pb.inc(1);

    // Memory benchmarks
    pb.set_message("Memory benchmarks...");
    let memory_scenario = if quick {
        MemoryScenario {
            num_allocations: 100,
            ..Default::default()
        }
    } else {
        MemoryScenario::default()
    };

    if let Ok(bench) = MemoryBenchmark::new(memory_scenario) {
        let results = bench.run();
        report = report.with_memory_results(results);
    }
    pb.inc(1);

    pb.finish_with_message("Complete!");
    println!();

    Ok(report)
}

fn run_transfer_benchmarks(iterations: usize, warmup: usize) -> anyhow::Result<BenchmarkReport> {
    println!("\n📊 Transfer Benchmarks\n");

    let scenario = TransferScenario {
        iterations,
        warmup,
        ..Default::default()
    };

    let bench = TransferBenchmark::new(scenario)?;
    let results = bench.run();

    let report = BenchmarkReport::new("Aether Transfer Benchmarks")
        .with_transfer_results(results);

    Ok(report)
}

fn run_decision_benchmarks(iterations: usize) -> anyhow::Result<BenchmarkReport> {
    println!("\n🧠 Decision Engine Benchmarks\n");

    let scenario = DecisionScenario {
        iterations,
        ..Default::default()
    };

    let bench = DecisionBenchmark::new(scenario);
    let results = bench.run();

    let report = BenchmarkReport::new("Aether Decision Engine Benchmarks")
        .with_decision_results(results);

    Ok(report)
}

fn run_memory_benchmarks(pool_size_gb: u64, num_allocations: usize) -> anyhow::Result<BenchmarkReport> {
    println!("\n💾 Memory Management Benchmarks\n");

    let scenario = MemoryScenario {
        pool_size: pool_size_gb * 1024 * 1024 * 1024,
        num_allocations,
        ..Default::default()
    };

    let bench = MemoryBenchmark::new(scenario)?;
    let results = bench.run();

    let report = BenchmarkReport::new("Aether Memory Management Benchmarks")
        .with_memory_results(results);

    Ok(report)
}

fn run_inference_benchmark(
    model: ModelConfig,
    memory_limit: Option<u64>,
) -> anyhow::Result<BenchmarkReport> {
    println!("\n🔮 Inference Benchmark: {}\n", model.name);

    // This would be a full inference benchmark
    // For now, just report model characteristics
    let report = BenchmarkReport::new(&format!("Aether Inference Benchmark: {}", model.name));

    println!("Model Configuration:");
    println!("  Layers: {}", model.num_layers);
    println!("  Hidden Dim: {}", model.hidden_dim);
    println!("  Heads: {} (KV: {})", model.num_heads, model.num_kv_heads);
    println!("  Vocab Size: {}", model.vocab_size);
    println!("  Model Size: {:.1} GB", model.model_size() as f64 / 1e9);
    println!("  KV Cache (bs=1, seq=2048): {:.1} MB",
             model.kv_cache_size(1, 2048) as f64 / 1e6);

    if model.is_moe {
        println!("  MoE: {} experts, top-{}",
                 model.num_experts.unwrap_or(0),
                 model.top_k_experts.unwrap_or(0));
    }

    if let Some(limit) = memory_limit {
        println!("\n  Memory Limit: {} GB", limit);
        let model_size_gb = model.model_size() as f64 / 1e9;
        if model_size_gb > limit as f64 {
            println!("  ⚠️  Model exceeds memory limit - offloading required");
        }
    }

    println!("\n  (Full inference benchmark not yet implemented)");

    Ok(report)
}

fn print_available_benchmarks() {
    println!("\n📋 Available Benchmarks\n");
    println!("Transfer Benchmarks:");
    println!("  - Host-to-Device (H2D)");
    println!("  - Device-to-Host (D2H)");
    println!("  - Device-to-Device (D2D)");
    println!("  - Concurrent transfers");
    println!();
    println!("Decision Engine Benchmarks:");
    println!("  - Planning throughput");
    println!("  - Scaling with state count");
    println!("  - Memory pressure response");
    println!();
    println!("Memory Benchmarks:");
    println!("  - Direct allocation");
    println!("  - Pool allocation");
    println!("  - Mixed-size allocation");
    println!("  - Allocation churn");
    println!("  - Fragmentation");
    println!();
    println!("Inference Scenarios:");
    println!("  - LLaMA 7B");
    println!("  - LLaMA 70B");
    println!("  - Mixtral 8x7B");
    println!("  - DeepSeek V2");
}
