//! s1-irrlab: a proxy that asks a System One decision model how complex a
//! prompt is and forwards the request to the top or the flash model.

pub mod anthropic;
pub mod config;
pub mod decider;
pub mod error;
pub mod openai;
pub mod server;
pub mod upstream;
