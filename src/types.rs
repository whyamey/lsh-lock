use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
pub enum AnalysisMode {
    Single,
    Pairs,
    Both,
}

impl FromStr for AnalysisMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "single" => Ok(AnalysisMode::Single),
            "pairs" => Ok(AnalysisMode::Pairs),
            "both" => Ok(AnalysisMode::Both),
            _ => Err(format!(
                "Unknown analysis mode: {}. Use 'single', 'pairs', or 'both'",
                s
            )),
        }
    }
}

impl Default for AnalysisMode {
    fn default() -> Self {
        AnalysisMode::Single
    }
}
