use std::collections::HashSet;

use serde::Serialize;

const MIN_TOKEN_BUDGET: usize = 1_000;
const MAX_TOKEN_BUDGET: usize = 30_000;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum ContextTaskType {
    Bugfix,
    Feature,
    Refactor,
    Review,
    Docs,
    Search,
    Planning,
    Maintenance,
}

impl ContextTaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bugfix => "bugfix",
            Self::Feature => "feature",
            Self::Refactor => "refactor",
            Self::Review => "review",
            Self::Docs => "docs",
            Self::Search => "search",
            Self::Planning => "planning",
            Self::Maintenance => "maintenance",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextSourcePlan {
    pub memories: bool,
    pub core_memories: bool,
    pub memory_graph: bool,
    pub code_index: bool,
    pub code_neighborhood: bool,
    pub code_memories: bool,
    pub eval_history: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextBudgetPlan {
    pub requested_token_budget: usize,
    pub effective_token_budget: usize,
    pub memory_tokens: usize,
    pub code_tokens: usize,
    pub graph_tokens: usize,
    pub code_memory_tokens: usize,
    pub response_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextPlan {
    pub task_type: String,
    pub query_terms: Vec<String>,
    pub source_plan: ContextSourcePlan,
    pub budget_plan: ContextBudgetPlan,
    pub memory_limit: usize,
    pub core_memory_limit: usize,
    pub code_limit: usize,
    pub graph_limit: usize,
    pub code_memory_limit: usize,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ContextPlanRequest<'a> {
    pub query: &'a str,
    pub memory_limit: usize,
    pub core_memory_limit: usize,
    pub code_limit: usize,
    pub token_budget: usize,
}

pub fn plan_context_access(request: ContextPlanRequest<'_>) -> ContextPlan {
    let terms = extract_query_terms(request.query);
    let task_type = classify_task(request.query, &terms);
    let source_plan = source_plan_for_task(&task_type);
    let effective_token_budget = request
        .token_budget
        .clamp(MIN_TOKEN_BUDGET, MAX_TOKEN_BUDGET);

    let mut memory_limit = request.memory_limit.clamp(1, 30);
    let mut core_memory_limit = request.core_memory_limit.clamp(0, 10);
    let mut code_limit = request.code_limit.clamp(0, 30);
    let mut reasons = vec![format!("classified task as {}", task_type.as_str())];

    match task_type {
        ContextTaskType::Bugfix => {
            memory_limit = memory_limit.min(8);
            core_memory_limit = core_memory_limit.min(4);
            code_limit = code_limit.clamp(10, 18);
            reasons.push(
                "bugfix tasks prioritize code hits, callers, callees, and nearby code memories"
                    .to_string(),
            );
        }
        ContextTaskType::Feature => {
            memory_limit = memory_limit.clamp(8, 14);
            core_memory_limit = core_memory_limit.clamp(3, 6);
            code_limit = code_limit.clamp(8, 16);
            reasons.push("feature tasks need balanced project rules, prior decisions, and implementation patterns".to_string());
        }
        ContextTaskType::Refactor => {
            memory_limit = memory_limit.min(8);
            core_memory_limit = core_memory_limit.min(4);
            code_limit = code_limit.clamp(12, 22);
            reasons.push(
                "refactor tasks prioritize code graph coverage and impact navigation".to_string(),
            );
        }
        ContextTaskType::Review => {
            memory_limit = memory_limit.clamp(6, 12);
            core_memory_limit = core_memory_limit.clamp(3, 6);
            code_limit = code_limit.clamp(10, 20);
            reasons.push(
                "review tasks need code, policy memories, and retrieval diagnostics".to_string(),
            );
        }
        ContextTaskType::Docs => {
            memory_limit = memory_limit.clamp(8, 16);
            core_memory_limit = core_memory_limit.clamp(4, 8);
            code_limit = code_limit.min(6);
            reasons.push(
                "documentation tasks prioritize durable decisions and only targeted code"
                    .to_string(),
            );
        }
        ContextTaskType::Search => {
            memory_limit = memory_limit.clamp(10, 20);
            core_memory_limit = core_memory_limit.min(4);
            code_limit = code_limit.clamp(6, 12);
            reasons
                .push("search tasks broaden memory recall while keeping code bounded".to_string());
        }
        ContextTaskType::Planning => {
            memory_limit = memory_limit.clamp(10, 18);
            core_memory_limit = core_memory_limit.clamp(4, 8);
            code_limit = code_limit.clamp(8, 16);
            reasons.push(
                "planning tasks need cross-source context before narrowing to files".to_string(),
            );
        }
        ContextTaskType::Maintenance => {
            memory_limit = memory_limit.clamp(8, 14);
            core_memory_limit = core_memory_limit.clamp(3, 6);
            code_limit = code_limit.clamp(8, 16);
            reasons.push(
                "maintenance tasks combine operational memory, eval state, and code index state"
                    .to_string(),
            );
        }
    }

    if !source_plan.code_index {
        code_limit = 0;
    }
    if !source_plan.memories {
        memory_limit = 1;
        core_memory_limit = 0;
    }

    let graph_limit = if source_plan.memory_graph {
        memory_limit.max(code_limit).clamp(8, 40)
    } else {
        0
    };
    let code_memory_limit = if source_plan.code_memories {
        code_limit.clamp(6, 24)
    } else {
        0
    };
    let budget_plan = split_budget(&task_type, effective_token_budget);

    ContextPlan {
        task_type: task_type.as_str().to_string(),
        query_terms: terms,
        source_plan,
        budget_plan,
        memory_limit,
        core_memory_limit,
        code_limit,
        graph_limit,
        code_memory_limit,
        reasons,
    }
}

pub fn estimate_context_tokens(parts: &[usize]) -> usize {
    parts.iter().copied().sum::<usize>().div_ceil(4)
}

fn source_plan_for_task(task_type: &ContextTaskType) -> ContextSourcePlan {
    match task_type {
        ContextTaskType::Docs => ContextSourcePlan {
            memories: true,
            core_memories: true,
            memory_graph: true,
            code_index: true,
            code_neighborhood: false,
            code_memories: true,
            eval_history: false,
        },
        ContextTaskType::Search => ContextSourcePlan {
            memories: true,
            core_memories: true,
            memory_graph: true,
            code_index: true,
            code_neighborhood: false,
            code_memories: true,
            eval_history: false,
        },
        ContextTaskType::Maintenance => ContextSourcePlan {
            memories: true,
            core_memories: true,
            memory_graph: true,
            code_index: true,
            code_neighborhood: true,
            code_memories: true,
            eval_history: true,
        },
        _ => ContextSourcePlan {
            memories: true,
            core_memories: true,
            memory_graph: true,
            code_index: true,
            code_neighborhood: true,
            code_memories: true,
            eval_history: false,
        },
    }
}

fn split_budget(task_type: &ContextTaskType, effective_token_budget: usize) -> ContextBudgetPlan {
    let (memory_pct, code_pct, graph_pct, code_memory_pct) = match task_type {
        ContextTaskType::Bugfix => (24, 48, 12, 10),
        ContextTaskType::Feature => (34, 36, 12, 12),
        ContextTaskType::Refactor => (20, 54, 12, 8),
        ContextTaskType::Review => (28, 44, 12, 10),
        ContextTaskType::Docs => (52, 24, 12, 6),
        ContextTaskType::Search => (50, 26, 12, 6),
        ContextTaskType::Planning => (40, 34, 12, 8),
        ContextTaskType::Maintenance => (34, 34, 14, 10),
    };
    let response_pct = 100usize
        .saturating_sub(memory_pct)
        .saturating_sub(code_pct)
        .saturating_sub(graph_pct)
        .saturating_sub(code_memory_pct);
    ContextBudgetPlan {
        requested_token_budget: effective_token_budget,
        effective_token_budget,
        memory_tokens: effective_token_budget * memory_pct / 100,
        code_tokens: effective_token_budget * code_pct / 100,
        graph_tokens: effective_token_budget * graph_pct / 100,
        code_memory_tokens: effective_token_budget * code_memory_pct / 100,
        response_tokens: effective_token_budget * response_pct / 100,
    }
}

fn classify_task(query: &str, terms: &[String]) -> ContextTaskType {
    let lower = query.to_ascii_lowercase();
    if has_any(
        &lower,
        terms,
        &[
            "bug",
            "fix",
            "error",
            "panic",
            "regression",
            "fail",
            "broken",
        ],
    ) {
        ContextTaskType::Bugfix
    } else if has_any(
        &lower,
        terms,
        &["review", "audit", "risk", "security", "проверь"],
    ) {
        ContextTaskType::Review
    } else if has_any(
        &lower,
        terms,
        &[
            "refactor",
            "rename",
            "cleanup",
            "simplify",
            "optimize",
            "оптимиз",
        ],
    ) {
        ContextTaskType::Refactor
    } else if has_any(
        &lower,
        terms,
        &["doc", "readme", "guide", "manual", "документ"],
    ) {
        ContextTaskType::Docs
    } else if has_any(
        &lower,
        terms,
        &["search", "find", "where", "locate", "найди", "поищи"],
    ) {
        ContextTaskType::Search
    } else if has_any(
        &lower,
        terms,
        &["plan", "roadmap", "design", "architecture", "план"],
    ) {
        ContextTaskType::Planning
    } else if has_any(
        &lower,
        terms,
        &[
            "maintenance",
            "backup",
            "eval",
            "embedding",
            "index",
            "cleanup",
        ],
    ) {
        ContextTaskType::Maintenance
    } else {
        ContextTaskType::Feature
    }
}

fn has_any(lower: &str, terms: &[String], needles: &[&str]) -> bool {
    needles
        .iter()
        .any(|needle| lower.contains(needle) || terms.iter().any(|term| term == needle))
}

fn extract_query_terms(query: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    query
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .map(str::trim)
        .filter(|term| term.chars().count() >= 3)
        .map(str::to_ascii_lowercase)
        .filter(|term| seen.insert(term.clone()))
        .take(24)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_prioritizes_code_for_bugfix() {
        let plan = plan_context_access(ContextPlanRequest {
            query: "fix panic in code symbol embedding cache",
            memory_limit: 8,
            core_memory_limit: 5,
            code_limit: 8,
            token_budget: 3_000,
        });
        assert_eq!(plan.task_type, "bugfix");
        assert!(plan.code_limit > plan.memory_limit);
        assert!(plan.source_plan.code_neighborhood);
    }

    #[test]
    fn planner_keeps_docs_memory_heavy() {
        let plan = plan_context_access(ContextPlanRequest {
            query: "update README docs for memory lifecycle",
            memory_limit: 8,
            core_memory_limit: 5,
            code_limit: 8,
            token_budget: 3_000,
        });
        assert_eq!(plan.task_type, "docs");
        assert!(plan.budget_plan.memory_tokens > plan.budget_plan.code_tokens);
        assert!(!plan.source_plan.code_neighborhood);
    }
}
