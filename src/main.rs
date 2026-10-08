use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use s1_irrlab::{config::Config, server};
use tracing::info;

/// Complexity-routing proxy for OpenAI- and Anthropic-compatible agents.
#[derive(Parser)]
#[command(name = "s1", version)]
struct Args {
    /// Path to the TOML configuration.
    #[arg(short, long, env = "S1_CONFIG", default_value = "s1.toml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cfg = Config::load(&args.config)?;
    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    info!(tier = "top", url = %cfg.top.base_url, model = %cfg.top.model, api_key = cfg.top.api_key.is_some());
    info!(tier = "flash", url = %cfg.flash.base_url, model = %cfg.flash.model, api_key = cfg.flash.api_key.is_some());
    info!(decider = %cfg.decider.url, threshold = cfg.decider.threshold, on_error = ?cfg.decider.on_error);

    let app = server::router(cfg)?;
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("cannot listen on {addr}"))?;
    info!("listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
