use anyhow::{Result, bail};
use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SafetyFinding {
    pub kind: String,
    pub severity: String,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SafetyReport {
    pub allowed: bool,
    pub findings: Vec<SafetyFinding>,
}

impl SafetyReport {
    pub fn blocking_findings(&self) -> Vec<&SafetyFinding> {
        self.findings
            .iter()
            .filter(|finding| finding.severity == "block")
            .collect()
    }
}

pub fn inspect_memory_safety(body: &str, tags: &[String], source: Option<&str>) -> SafetyReport {
    let mut findings = Vec::new();
    inspect_text("body", body, &mut findings);
    for tag in tags {
        inspect_text("tag", tag, &mut findings);
    }
    if let Some(source) = source {
        inspect_text("source", source, &mut findings);
    }
    deduplicate_findings(&mut findings);
    let allowed = !findings.iter().any(|finding| finding.severity == "block");
    SafetyReport { allowed, findings }
}

pub fn enforce_memory_safety(report: &SafetyReport, allow_sensitive: bool) -> Result<()> {
    if allow_sensitive || report.allowed {
        return Ok(());
    }
    let details = report
        .blocking_findings()
        .into_iter()
        .map(|finding| format!("{}: {}", finding.kind, finding.evidence))
        .collect::<Vec<_>>()
        .join("; ");
    bail!(
        "memory safety policy blocked write: {details}. Use allow_sensitive only for explicit manual recovery."
    )
}

pub fn looks_like_sensitive_text(text: &str) -> bool {
    !inspect_memory_safety(text, &[], None).allowed
}

fn inspect_text(field: &str, text: &str, findings: &mut Vec<SafetyFinding>) {
    let lower = text.to_ascii_lowercase();
    if text.contains("-----BEGIN ") {
        findings.push(block("private_key", field, "PEM private key marker"));
    }
    for prefix in [
        "ghp_",
        "gho_",
        "github_pat_",
        "xoxb-",
        "xoxp-",
        "ya29.",
        "sk-",
        "AKIA",
    ] {
        if contains_token_prefix(text, prefix) {
            findings.push(block("token_prefix", field, prefix));
        }
    }
    for label in [
        "api_key",
        "apikey",
        "api key",
        "access token",
        "auth token",
        "bearer",
        "password",
        "passwd",
        "secret",
        "private key",
    ] {
        if contains_secret_assignment(&lower, label) {
            findings.push(block("secret_assignment", field, label));
        }
    }
    if contains_labeled_high_entropy_token(&lower, text) {
        findings.push(block(
            "high_entropy_secret",
            field,
            "credential-like long token near a secret label",
        ));
    }
    if contains_email_like(text) {
        findings.push(warn("possible_email", field, "email-like text"));
    }
    if contains_phone_like(text) {
        findings.push(warn("possible_phone", field, "phone-like digit sequence"));
    }
}

fn contains_secret_assignment(lower: &str, label: &str) -> bool {
    let Some(index) = lower.find(label) else {
        return false;
    };
    let tail = lower[index + label.len()..].trim_start();
    tail.starts_with('=')
        || tail.starts_with(':')
        || tail.starts_with(" is ")
        || tail.starts_with(" =")
        || tail.starts_with(" :")
}

fn contains_labeled_high_entropy_token(lower: &str, original: &str) -> bool {
    let has_label = [
        "api",
        "token",
        "password",
        "passwd",
        "secret",
        "credential",
        "key",
    ]
    .iter()
    .any(|label| lower.contains(label));
    has_label
        && original
            .split(|ch: char| ch.is_whitespace() || matches!(ch, '"' | '\'' | ',' | ';'))
            .any(is_high_entropy_token)
}

fn contains_token_prefix(text: &str, prefix: &str) -> bool {
    text.split(|ch: char| ch.is_whitespace() || matches!(ch, '"' | '\'' | ',' | ';' | ':' | '='))
        .map(|part| part.trim_matches(|ch: char| matches!(ch, '[' | ']' | '(' | ')' | '<' | '>')))
        .any(|part| part.starts_with(prefix) && part.len() >= prefix.len() + 8)
}

fn is_high_entropy_token(token: &str) -> bool {
    let token = token.trim_matches(|ch: char| matches!(ch, '=' | ':' | '[' | ']' | '(' | ')'));
    if token.len() < 32 {
        return false;
    }
    let letters = token.chars().filter(|ch| ch.is_ascii_alphabetic()).count();
    let digits = token.chars().filter(|ch| ch.is_ascii_digit()).count();
    let symbols = token
        .chars()
        .filter(|ch| matches!(ch, '_' | '-' | '.' | '/' | '+' | '='))
        .count();
    letters >= 8 && digits >= 4 && symbols >= 1
}

fn contains_email_like(text: &str) -> bool {
    text.split_whitespace().any(|part| {
        let part = part.trim_matches(|ch: char| {
            matches!(ch, ',' | ';' | ':' | ')' | '(' | '[' | ']' | '<' | '>')
        });
        let Some((left, right)) = part.split_once('@') else {
            return false;
        };
        left.len() >= 2 && right.contains('.') && right.len() >= 4
    })
}

fn contains_phone_like(text: &str) -> bool {
    let digits = text.chars().filter(|ch| ch.is_ascii_digit()).count();
    digits >= 10 && text.chars().any(|ch| matches!(ch, '+' | '(' | ')' | '-'))
}

fn block(kind: &str, field: &str, evidence: &str) -> SafetyFinding {
    SafetyFinding {
        kind: kind.to_string(),
        severity: "block".to_string(),
        evidence: format!("{field}: {evidence}"),
    }
}

fn warn(kind: &str, field: &str, evidence: &str) -> SafetyFinding {
    SafetyFinding {
        kind: kind.to_string(),
        severity: "warn".to_string(),
        evidence: format!("{field}: {evidence}"),
    }
}

fn deduplicate_findings(findings: &mut Vec<SafetyFinding>) {
    let mut seen = std::collections::HashSet::new();
    findings.retain(|finding| {
        seen.insert((
            finding.kind.clone(),
            finding.severity.clone(),
            finding.evidence.clone(),
        ))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_explicit_secret_assignment() {
        let report = inspect_memory_safety("API key: sk-test1234567890abcdef", &[], None);
        assert!(!report.allowed);
        assert!(
            report
                .findings
                .iter()
                .any(|finding| finding.severity == "block")
        );
    }

    #[test]
    fn warns_but_allows_email_like_text() {
        let report =
            inspect_memory_safety("Contact owner@example.com for asset approvals.", &[], None);
        assert!(report.allowed);
        assert!(
            report
                .findings
                .iter()
                .any(|finding| finding.kind == "possible_email")
        );
    }

    #[test]
    fn allows_policy_text_about_passwords_without_secret_value() {
        let report = inspect_memory_safety(
            "Never store user passwords in project memory; use a reference to the vault.",
            &[],
            None,
        );
        assert!(report.allowed);
    }
}
