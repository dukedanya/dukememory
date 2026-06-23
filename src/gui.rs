use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::f32::consts::TAU;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::sync::mpsc;
use std::thread;

use anyhow::{Result, anyhow, bail};
use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, RichText, ScrollArea, Sense, Stroke, Vec2,
};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::graph_extract::{GraphExtractionReport, apply_graph_extraction, extract_memory_graph};
use crate::ollama::OllamaClient;
use crate::store::{
    CodeRelation, CodeSearchOptions, CodeSymbol, ListOptions, Memory, MemoryGraph, MemoryStatus,
    ProjectRecord, ProjectStatus, SearchOptions, StatusFilter, Store,
};

const MIN_GRAPH_ZOOM: f32 = 0.46;
const MAX_GRAPH_ZOOM: f32 = 3.0;
const NODE_COLLISION_PADDING: f32 = 18.0;

pub fn run_memory_viewer(
    config: &Config,
    project_id: String,
    initial_status: String,
    initial_kind: Option<String>,
    limit: usize,
) -> Result<()> {
    let status = MemoryStatusChoice::parse(&initial_status)?;
    let store = Store::open(&config.database_marker)?;
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1240.0, 800.0])
            .with_min_inner_size([940.0, 620.0]),
        ..Default::default()
    };
    let app = MemoryViewerApp::new(
        store,
        project_id,
        config.clone(),
        config.database_marker.clone(),
        status,
        initial_kind.unwrap_or_default(),
        limit.clamp(1, 500),
    );

    eframe::run_native(
        "Dukememory",
        native_options,
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(obsidian_visuals());
            Ok(Box::new(app))
        }),
    )
    .map_err(|error| anyhow!("failed to run native memory viewer: {error}"))
}

fn obsidian_visuals() -> egui::Visuals {
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(Color32::from_rgb(221, 218, 232));
    visuals.panel_fill = Color32::from_rgb(30, 30, 38);
    visuals.window_fill = Color32::from_rgb(33, 32, 41);
    visuals.faint_bg_color = Color32::from_rgb(38, 37, 48);
    visuals.extreme_bg_color = Color32::from_rgb(22, 22, 29);
    visuals.code_bg_color = Color32::from_rgb(42, 40, 54);
    visuals.hyperlink_color = Color32::from_rgb(184, 167, 255);
    visuals.warn_fg_color = Color32::from_rgb(227, 157, 74);
    visuals.error_fg_color = Color32::from_rgb(232, 92, 92);
    visuals.selection.bg_fill = Color32::from_rgb(92, 75, 153);
    visuals.selection.stroke = Stroke::new(1.0, Color32::from_rgb(190, 174, 255));
    visuals.window_stroke = Stroke::new(1.0, Color32::from_rgb(58, 55, 72));
    visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(30, 30, 38);
    visuals.widgets.noninteractive.weak_bg_fill = Color32::from_rgb(30, 30, 38);
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(58, 55, 72));
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, Color32::from_rgb(177, 174, 190));
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(45, 43, 56);
    visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(43, 41, 53);
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(65, 62, 80));
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, Color32::from_rgb(217, 214, 229));
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(58, 54, 78);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(54, 50, 70);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(120, 103, 196));
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.2, Color32::from_rgb(239, 236, 250));
    visuals.widgets.active.bg_fill = Color32::from_rgb(74, 63, 118);
    visuals.widgets.active.weak_bg_fill = Color32::from_rgb(74, 63, 118);
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, Color32::from_rgb(190, 174, 255));
    visuals.widgets.active.fg_stroke = Stroke::new(1.4, Color32::WHITE);
    visuals.widgets.open.bg_fill = Color32::from_rgb(39, 38, 49);
    visuals.widgets.open.weak_bg_fill = Color32::from_rgb(42, 40, 53);
    visuals.widgets.open.bg_stroke = Stroke::new(1.0, Color32::from_rgb(70, 66, 88));
    visuals.widgets.open.fg_stroke = Stroke::new(1.0, Color32::from_rgb(226, 222, 238));
    visuals.striped = true;
    visuals
}

struct MemoryViewerApp {
    store: Store,
    project_id: String,
    projects: Vec<ProjectRecord>,
    project_note: Option<String>,
    database_marker: PathBuf,
    ollama_base_url: String,
    ollama_llm_model: String,
    status: MemoryStatusChoice,
    kind: String,
    query: String,
    node_filter: String,
    relation_filter: String,
    code_kind_filter: String,
    code_file_filter: String,
    limit: usize,
    memories: Vec<Memory>,
    graph: MemoryGraph,
    code_symbols: Vec<CodeSymbol>,
    code_relations: Vec<CodeRelation>,
    data_revision: u64,
    model_cache: Option<ModelCache>,
    layout_cache: Option<LayoutCache>,
    graph_pan: Vec2,
    graph_zoom: f32,
    graph_node_offsets: BTreeMap<String, Vec2>,
    pinned_nodes: BTreeSet<String>,
    fit_requested: bool,
    selected: GraphSelection,
    selection_back: Vec<GraphSelection>,
    selection_forward: Vec<GraphSelection>,
    expanded_cluster: Option<String>,
    map_domain: GraphDomain,
    view_mode: GraphViewMode,
    show_facts: bool,
    show_evidence: bool,
    show_minimap: bool,
    show_legend: bool,
    show_quality: bool,
    focus_selected: bool,
    max_visible_nodes: usize,
    project_status: Option<ProjectStatus>,
    project_root: Option<PathBuf>,
    settings_path: PathBuf,
    project_views: BTreeMap<String, SavedProjectView>,
    extract_query: String,
    extract_limit: usize,
    extract_apply: bool,
    extract_running: bool,
    extract_receiver: Option<mpsc::Receiver<Result<GraphExtractionReport, String>>>,
    extract_report: Option<GraphExtractionReport>,
    error: Option<String>,
    graph_note: Option<String>,
    code_note: Option<String>,
    last_refresh: String,
}

impl MemoryViewerApp {
    fn new(
        store: Store,
        project_id: String,
        config: Config,
        database_marker: PathBuf,
        status: MemoryStatusChoice,
        kind: String,
        limit: usize,
    ) -> Self {
        let project_root = store
            .project_profile(&project_id)
            .ok()
            .and_then(|profile| profile.root_path.map(PathBuf::from));
        let projects = store.list_projects().unwrap_or_default();
        let settings_path = gui_settings_path(&database_marker);
        let mut app = Self {
            store,
            project_id,
            projects,
            project_note: None,
            database_marker,
            ollama_base_url: config.ollama_base_url,
            ollama_llm_model: config.ollama_llm_model,
            status,
            kind,
            query: String::new(),
            node_filter: String::new(),
            relation_filter: String::new(),
            code_kind_filter: String::new(),
            code_file_filter: String::new(),
            limit,
            memories: Vec::new(),
            graph: empty_graph(),
            code_symbols: Vec::new(),
            code_relations: Vec::new(),
            data_revision: 0,
            model_cache: None,
            layout_cache: None,
            graph_pan: Vec2::ZERO,
            graph_zoom: 1.0,
            graph_node_offsets: BTreeMap::new(),
            pinned_nodes: BTreeSet::new(),
            fit_requested: true,
            selected: GraphSelection::Overview,
            selection_back: Vec::new(),
            selection_forward: Vec::new(),
            expanded_cluster: None,
            map_domain: GraphDomain::Memory,
            view_mode: GraphViewMode::Overview,
            show_facts: false,
            show_evidence: false,
            show_minimap: true,
            show_legend: true,
            show_quality: true,
            focus_selected: true,
            max_visible_nodes: 80,
            project_status: None,
            project_root,
            settings_path,
            project_views: BTreeMap::new(),
            extract_query: String::new(),
            extract_limit: 20,
            extract_apply: false,
            extract_running: false,
            extract_receiver: None,
            extract_report: None,
            error: None,
            graph_note: None,
            code_note: None,
            last_refresh: String::new(),
        };
        app.load_settings();
        app.refresh();
        app
    }

    fn refresh(&mut self) {
        let kind = self.normalized_kind();
        let status = self.status.filter();
        let memories = if self.query.trim().is_empty() {
            self.store.list(
                &self.project_id,
                ListOptions {
                    limit: self.limit,
                    offset: 0,
                    status,
                    kind,
                    memory_tier: None,
                },
            )
        } else {
            self.store.search(
                &self.project_id,
                SearchOptions {
                    query: self.query.trim().to_string(),
                    limit: self.limit,
                    status,
                    kind,
                    memory_tier: None,
                },
            )
        };

        match memories {
            Ok(memories) => {
                self.memories = memories;
                self.error = None;
            }
            Err(error) => {
                self.error = Some(error.to_string());
                return;
            }
        }

        match self
            .store
            .search_memory_graph(&self.project_id, self.query.trim(), self.limit)
        {
            Ok(graph) => {
                self.graph = graph;
                self.graph_note = None;
            }
            Err(error) => {
                self.graph = empty_graph();
                self.graph_note = Some(format!("графовые связи не загрузились: {error}"));
            }
        }

        let code_symbols = if self.query.trim().is_empty() {
            self.store.code_symbols_for_project(&self.project_id)
        } else {
            self.store
                .search_code(
                    &self.project_id,
                    CodeSearchOptions {
                        query: self.query.trim().to_string(),
                        limit: self.limit.clamp(1, 500),
                        kind: self.normalized_code_kind(),
                        file_path: None,
                    },
                )
                .map(|results| results.into_iter().map(|result| result.symbol).collect())
        };

        match code_symbols {
            Ok(mut symbols) => {
                self.apply_code_filters(&mut symbols);
                symbols.truncate(self.limit.clamp(1, 500));
                let seed_ids = symbols
                    .iter()
                    .map(|symbol| symbol.id.clone())
                    .collect::<Vec<_>>();
                match self.store.code_graph_for_symbols(
                    &self.project_id,
                    &seed_ids,
                    self.limit.clamp(1, 500),
                ) {
                    Ok((related_symbols, relations)) => {
                        self.code_symbols = if related_symbols.is_empty() {
                            symbols
                        } else {
                            related_symbols
                        };
                        self.code_relations = relations;
                        self.code_note = None;
                    }
                    Err(error) => {
                        self.code_symbols = symbols;
                        self.code_relations.clear();
                        self.code_note = Some(format!("связи кода не загрузились: {error}"));
                    }
                }
            }
            Err(error) => {
                self.code_symbols.clear();
                self.code_relations.clear();
                self.code_note = Some(format!("граф кода не загрузился: {error}"));
            }
        }

        self.last_refresh = format!(
            "память {}, код {}",
            self.memories.len(),
            self.code_symbols.len()
        );

        match self.store.status(&self.project_id) {
            Ok(status) => self.project_status = Some(status),
            Err(error) => self.error = Some(error.to_string()),
        }
        self.refresh_projects();
        self.data_revision = self.data_revision.wrapping_add(1);
        self.invalidate_graph_caches();
    }

    fn refresh_projects(&mut self) {
        match self.store.list_projects() {
            Ok(projects) => {
                self.projects = projects;
                self.project_note = None;
            }
            Err(error) => {
                self.project_note = Some(format!("проекты не загрузились: {error}"));
            }
        }
    }

    fn current_project(&self) -> Option<&ProjectRecord> {
        self.projects
            .iter()
            .find(|project| project.id == self.project_id)
    }

    fn switch_project(&mut self, project_id: String) {
        if self.project_id == project_id {
            return;
        }

        self.save_settings();
        self.project_id = project_id;
        self.project_root = self
            .store
            .project_profile(&self.project_id)
            .ok()
            .and_then(|profile| profile.root_path.map(PathBuf::from));
        self.memories.clear();
        self.graph = empty_graph();
        self.code_symbols.clear();
        self.code_relations.clear();
        self.project_status = None;
        self.selected = GraphSelection::Overview;
        self.selection_back.clear();
        self.selection_forward.clear();
        self.graph_pan = Vec2::ZERO;
        self.graph_zoom = 1.0;
        self.graph_node_offsets.clear();
        self.pinned_nodes.clear();
        self.fit_requested = true;
        self.expanded_cluster = None;
        self.extract_running = false;
        self.extract_receiver = None;
        self.extract_report = None;
        self.error = None;
        self.graph_note = None;
        self.code_note = None;
        self.last_refresh = String::new();
        self.data_revision = self.data_revision.wrapping_add(1);
        self.load_current_project_view();
        self.invalidate_graph_caches();
        self.refresh();
    }

    fn normalized_kind(&self) -> Option<String> {
        let kind = self.kind.trim();
        if kind.is_empty() {
            None
        } else {
            Some(kind.to_string())
        }
    }

    fn normalized_code_kind(&self) -> Option<String> {
        let kind = self.code_kind_filter.trim();
        if kind.is_empty() {
            None
        } else {
            Some(kind.to_string())
        }
    }

    fn normalized_code_file(&self) -> Option<String> {
        let file_path = self.code_file_filter.trim();
        if file_path.is_empty() {
            None
        } else {
            Some(file_path.to_ascii_lowercase())
        }
    }

    fn apply_code_filters(&self, symbols: &mut Vec<CodeSymbol>) {
        if self.query.trim().is_empty()
            && let Some(kind) = self.normalized_code_kind()
        {
            symbols.retain(|symbol| symbol.kind == kind);
        }
        if let Some(file_filter) = self.normalized_code_file() {
            symbols.retain(|symbol| symbol.file_path.to_ascii_lowercase().contains(&file_filter));
        }
    }

    fn invalidate_graph_caches(&mut self) {
        self.model_cache = None;
        self.layout_cache = None;
    }

    fn reset_graph_view(&mut self) {
        self.graph_pan = Vec2::ZERO;
        self.graph_zoom = 1.0;
        self.fit_requested = true;
    }

    fn save_current_project_view(&mut self) {
        self.project_views.insert(
            self.project_id.clone(),
            SavedProjectView::from_app(
                self.graph_pan,
                self.graph_zoom,
                &self.graph_node_offsets,
                &self.pinned_nodes,
                self.map_domain,
                self.view_mode,
                self.show_facts,
                self.show_evidence,
                self.focus_selected,
                self.max_visible_nodes,
                self.expanded_cluster.clone(),
            ),
        );
    }

    fn load_current_project_view(&mut self) {
        let Some(view) = self.project_views.get(&self.project_id).cloned() else {
            self.reset_graph_view();
            self.pinned_nodes.clear();
            self.expanded_cluster = None;
            return;
        };
        self.graph_pan = view.pan.to_vec2();
        self.graph_zoom = view.zoom.clamp(MIN_GRAPH_ZOOM, MAX_GRAPH_ZOOM);
        self.graph_node_offsets = view
            .node_offsets
            .into_iter()
            .map(|(id, offset)| (id, offset.to_vec2()))
            .collect();
        self.pinned_nodes = view.pinned_nodes.into_iter().collect();
        self.map_domain = view.map_domain;
        self.view_mode = view.view_mode;
        self.show_facts = view.show_facts;
        self.show_evidence = view.show_evidence;
        self.focus_selected = view.focus_selected;
        self.max_visible_nodes = view.max_visible_nodes.clamp(20, 250);
        self.expanded_cluster = view.expanded_cluster;
        self.fit_requested = false;
        self.invalidate_graph_caches();
    }

    fn toggle_pin(&mut self, node_id: &str) {
        if self.pinned_nodes.remove(node_id) {
            self.graph_node_offsets.remove(node_id);
        } else {
            self.pinned_nodes.insert(node_id.to_string());
            self.graph_node_offsets
                .entry(node_id.to_string())
                .or_insert(Vec2::ZERO);
        }
        self.save_current_project_view();
    }

    fn open_graph_view(&mut self, domain: GraphDomain, view_mode: GraphViewMode) {
        self.map_domain = domain;
        self.view_mode = view_mode;
        self.expanded_cluster = None;
        self.selected = GraphSelection::Overview;
        self.selection_back.clear();
        self.selection_forward.clear();
        self.graph_node_offsets.clear();
        self.pinned_nodes.clear();
        self.fit_requested = true;
        self.invalidate_graph_caches();
        self.save_current_project_view();
    }

    fn open_overview_area(&mut self, node_id: &str) -> bool {
        let Some(target) = overview_navigation_target(node_id) else {
            return false;
        };
        if target.show_facts {
            self.show_facts = true;
        }
        self.open_graph_view(target.domain, target.view_mode);
        true
    }

    fn select(&mut self, selection: GraphSelection) {
        if self.selected == selection {
            return;
        }
        self.selection_back.push(self.selected.clone());
        self.selection_forward.clear();
        self.selected = selection;
    }

    fn go_back(&mut self) {
        if let Some(previous) = self.selection_back.pop() {
            self.selection_forward.push(self.selected.clone());
            self.selected = previous;
        }
    }

    fn go_forward(&mut self) {
        if let Some(next) = self.selection_forward.pop() {
            self.selection_back.push(self.selected.clone());
            self.selected = next;
        }
    }

    fn center_selection(&mut self) {
        self.reset_graph_view();
        self.expanded_cluster = None;
        self.select(GraphSelection::Overview);
    }

    fn fit_graph_to_view(
        &mut self,
        rect: Rect,
        model: &GraphModel,
        positions: &BTreeMap<String, Pos2>,
    ) {
        let Some(bounds) = graph_world_bounds_for_model(
            model,
            positions,
            &self.graph_node_offsets,
            MIN_GRAPH_ZOOM,
        ) else {
            self.graph_zoom = 1.0;
            self.graph_pan = Vec2::ZERO;
            return;
        };
        let margin = 48.0;
        let available_width = (rect.width() - margin).max(240.0);
        let available_height = (rect.height() - margin).max(220.0);
        let zoom = (available_width / bounds.width().max(1.0))
            .min(available_height / bounds.height().max(1.0))
            .clamp(MIN_GRAPH_ZOOM, 2.4);
        self.graph_zoom = zoom;
        let center_delta: Vec2 = rect.center() - bounds.center();
        self.graph_pan = center_delta * zoom;
    }

    fn zoom_graph_at(&mut self, rect: Rect, focus: Pos2, zoom_delta: f32) {
        let old_zoom = self.graph_zoom.clamp(MIN_GRAPH_ZOOM, MAX_GRAPH_ZOOM);
        let new_zoom = (old_zoom * zoom_delta).clamp(MIN_GRAPH_ZOOM, MAX_GRAPH_ZOOM);
        if (new_zoom - old_zoom).abs() < 0.001 {
            return;
        }
        let centered_focus = focus - rect.center();
        let world_at_focus = (centered_focus - self.graph_pan) / old_zoom;
        self.graph_pan = centered_focus - world_at_focus * new_zoom;
        self.graph_zoom = new_zoom;
    }

    fn expand_cluster(&mut self, node_id: String) {
        self.expanded_cluster = Some(node_id);
        self.selected = GraphSelection::Overview;
        self.selection_back.clear();
        self.selection_forward.clear();
        self.graph_pan = Vec2::ZERO;
        self.graph_zoom = 1.0;
        self.graph_node_offsets.clear();
        self.pinned_nodes.clear();
        self.fit_requested = true;
        self.invalidate_graph_caches();
    }

    fn collapse_cluster(&mut self) {
        if self.expanded_cluster.is_none() {
            return;
        }
        self.expanded_cluster = None;
        self.selected = GraphSelection::Overview;
        self.selection_back.clear();
        self.selection_forward.clear();
        self.graph_pan = Vec2::ZERO;
        self.graph_zoom = 1.0;
        self.graph_node_offsets.clear();
        self.pinned_nodes.clear();
        self.fit_requested = true;
        self.invalidate_graph_caches();
    }

    fn reset_filters(&mut self) {
        self.kind.clear();
        self.query.clear();
        self.node_filter.clear();
        self.relation_filter.clear();
        self.code_kind_filter.clear();
        self.code_file_filter.clear();
        self.status = MemoryStatusChoice::Active;
        self.center_selection();
        self.refresh();
    }

    fn load_settings(&mut self) {
        let Ok(raw) = fs::read_to_string(&self.settings_path) else {
            return;
        };
        let Ok(settings) = serde_json::from_str::<GuiSettings>(&raw) else {
            return;
        };
        self.node_filter = settings.node_filter;
        self.relation_filter = settings.relation_filter;
        self.code_kind_filter = settings.code_kind_filter;
        self.code_file_filter = settings.code_file_filter;
        self.map_domain = settings.map_domain;
        self.view_mode = settings.view_mode;
        self.show_facts = settings.show_facts;
        self.show_evidence = settings.show_evidence;
        self.show_minimap = settings.show_minimap;
        self.show_legend = settings.show_legend;
        self.show_quality = settings.show_quality;
        self.focus_selected = settings.focus_selected;
        self.max_visible_nodes = settings.max_visible_nodes.clamp(20, 250);
        self.project_views = settings.project_views;
        self.load_current_project_view();
    }

    fn save_settings(&mut self) {
        self.save_current_project_view();
        let settings = GuiSettings {
            node_filter: self.node_filter.clone(),
            relation_filter: self.relation_filter.clone(),
            code_kind_filter: self.code_kind_filter.clone(),
            code_file_filter: self.code_file_filter.clone(),
            map_domain: self.map_domain,
            view_mode: self.view_mode,
            show_facts: self.show_facts,
            show_evidence: self.show_evidence,
            show_minimap: self.show_minimap,
            show_legend: self.show_legend,
            show_quality: self.show_quality,
            focus_selected: self.focus_selected,
            max_visible_nodes: self.max_visible_nodes,
            project_views: self.project_views.clone(),
        };
        let Ok(raw) = serde_json::to_string_pretty(&settings) else {
            return;
        };
        if let Some(parent) = self.settings_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&self.settings_path, raw);
    }

    fn export_current_map(&mut self) {
        let model = self.current_model();
        let path = std::env::temp_dir().join("dukememory-map.json");
        let result = serde_json::to_string_pretty(&model)
            .map_err(anyhow::Error::from)
            .and_then(|raw| fs::write(&path, raw).map_err(anyhow::Error::from));
        match result {
            Ok(()) => {
                self.last_refresh = format!("экспорт: {}", path.display());
                self.error = None;
            }
            Err(error) => self.error = Some(format!("экспорт не удался: {error}")),
        }
    }

    fn open_code_file(&mut self, file_path: &str) {
        let path = self
            .project_root
            .as_ref()
            .map(|root| root.join(file_path))
            .unwrap_or_else(|| PathBuf::from(file_path));
        if !path.exists() {
            self.error = Some(format!("файл не найден: {}", path.display()));
            return;
        }
        match ProcessCommand::new("open").arg(&path).spawn() {
            Ok(_) => self.error = None,
            Err(error) => self.error = Some(format!("не удалось открыть файл: {error}")),
        }
    }

    fn poll_graph_extract(&mut self) {
        let Some(receiver) = &self.extract_receiver else {
            return;
        };
        match receiver.try_recv() {
            Ok(Ok(report)) => {
                let applied = report.apply;
                self.extract_running = false;
                self.extract_receiver = None;
                self.last_refresh = format!(
                    "извлечение графа: {} сущн., {} фактов, {} связей",
                    report.proposed_entities, report.proposed_facts, report.proposed_edges
                );
                self.extract_report = Some(report);
                self.error = None;
                if applied {
                    self.refresh();
                }
            }
            Ok(Err(error)) => {
                self.extract_running = false;
                self.extract_receiver = None;
                self.error = Some(format!("извлечение графа не удалось: {error}"));
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.extract_running = false;
                self.extract_receiver = None;
                self.error = Some("извлечение графа остановилось без результата".to_string());
            }
        }
    }

    fn start_graph_extract(&mut self, apply: bool) {
        if self.extract_running {
            return;
        }
        let project_id = self.project_id.clone();
        let database_marker = self.database_marker.clone();
        let ollama_base_url = self.ollama_base_url.clone();
        let ollama_llm_model = self.ollama_llm_model.clone();
        let query = self.extract_query.trim().to_string();
        let limit = self.extract_limit.clamp(1, 100);
        let status = self.status;
        let kind = self.normalized_kind();
        let (sender, receiver) = mpsc::channel();
        self.extract_apply = apply;
        self.extract_running = true;
        self.extract_receiver = Some(receiver);
        self.extract_report = None;
        self.error = None;

        thread::spawn(move || {
            let result = run_graph_extract_job(GraphExtractJob {
                database_marker,
                ollama_base_url,
                ollama_llm_model,
                project_id,
                query,
                limit,
                status,
                kind,
                apply,
            })
            .map_err(|error| error.to_string());
            let _ = sender.send(result);
        });
    }

    fn current_model(&mut self) -> GraphModel {
        let key = self.model_cache_key();
        if let Some(cache) = &self.model_cache
            && cache.key == key
        {
            return cache.model.clone();
        }

        let model = self.relationship_model();
        self.model_cache = Some(ModelCache {
            key,
            model: model.clone(),
        });
        self.layout_cache = None;
        model
    }

    fn validated_model(&mut self) -> GraphModel {
        let mut model = self.current_model();
        if !model.contains_selection(&self.selected) {
            self.selected = GraphSelection::Overview;
            self.invalidate_graph_caches();
            model = self.current_model();
        }
        model
    }

    fn model_cache_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.data_revision.hash(&mut hasher);
        self.selected.hash(&mut hasher);
        self.map_domain.hash(&mut hasher);
        self.view_mode.hash(&mut hasher);
        self.show_facts.hash(&mut hasher);
        self.show_evidence.hash(&mut hasher);
        self.focus_selected.hash(&mut hasher);
        self.max_visible_nodes.hash(&mut hasher);
        self.node_filter.hash(&mut hasher);
        self.relation_filter.hash(&mut hasher);
        self.expanded_cluster.hash(&mut hasher);
        hasher.finish()
    }

    fn project_sidebar_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("Проекты");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("↻")
                    .on_hover_text("Обновить список")
                    .clicked()
                {
                    self.refresh_projects();
                }
            });
        });
        ui.separator();

        let projects = self.projects.clone();
        let current_id = self.project_id.clone();
        let mut next_project = None;

        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if projects.is_empty() {
                    ui.label("Проекты не найдены");
                    return;
                }

                for project in projects {
                    let selected = project.id == current_id;
                    let title = if project.name.trim().is_empty() {
                        project.id.as_str()
                    } else {
                        project.name.as_str()
                    };
                    let response = ui
                        .selectable_label(
                            selected,
                            RichText::new(truncate_text(title, 28)).strong(),
                        )
                        .on_hover_text(project.id.as_str());
                    if response.clicked() {
                        next_project = Some(project.id.clone());
                    }
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            RichText::new(project.project_type.as_str())
                                .small()
                                .color(Color32::from_rgb(166, 143, 255)),
                        );
                        ui.label(
                            RichText::new(truncate_text(project.id.as_str(), 24))
                                .small()
                                .monospace()
                                .color(Color32::from_rgb(142, 140, 153)),
                        );
                    });
                    if let Some(root_path) = &project.root_path {
                        ui.label(
                            RichText::new(truncate_text(root_path, 32))
                                .small()
                                .color(Color32::from_rgb(132, 130, 142)),
                        );
                    }
                    ui.add_space(7.0);
                }
            });

        if let Some(note) = &self.project_note {
            ui.separator();
            ui.colored_label(Color32::from_rgb(205, 146, 71), note);
        }

        if let Some(project_id) = next_project {
            self.switch_project(project_id);
        }
    }

    fn header_ui(&mut self, ui: &mut egui::Ui) {
        if self.map_domain != GraphDomain::Code && self.view_mode == GraphViewMode::Files {
            self.view_mode = GraphViewMode::Types;
        }
        let project_title = self
            .current_project()
            .map(|project| project.name.as_str())
            .filter(|name| !name.trim().is_empty())
            .unwrap_or(self.project_id.as_str())
            .to_string();

        ui.horizontal_wrapped(|ui| {
            ui.heading("Карта");
            ui.label(
                RichText::new(project_title)
                    .strong()
                    .color(Color32::from_rgb(202, 191, 255)),
            );
            ui.separator();
            let search_response = ui.add_sized(
                [420.0, 26.0],
                egui::TextEdit::singleline(&mut self.query).hint_text("Найти в карте"),
            );
            let enter_pressed = ui.input(|input| input.key_pressed(egui::Key::Enter));
            if ui.button("Найти").clicked() || (enter_pressed && search_response.lost_focus())
            {
                self.center_selection();
                self.refresh();
            }
            if ui
                .add_enabled(
                    !self.query.trim().is_empty()
                        || !self.node_filter.trim().is_empty()
                        || !self.relation_filter.trim().is_empty()
                        || !self.kind.trim().is_empty()
                        || !self.code_kind_filter.trim().is_empty()
                        || !self.code_file_filter.trim().is_empty(),
                    egui::Button::new("Очистить"),
                )
                .clicked()
            {
                self.reset_filters();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Обновить").clicked() {
                    self.refresh();
                }
            });
        });

        ui.add_space(6.0);
        let previous_domain = self.map_domain;
        let previous_view = self.view_mode;
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.map_domain, GraphDomain::Memory, "Память");
            ui.selectable_value(&mut self.map_domain, GraphDomain::Code, "Код");
            ui.selectable_value(&mut self.map_domain, GraphDomain::Bridge, "Связи");
            ui.separator();
            ui.selectable_value(&mut self.view_mode, GraphViewMode::Overview, "Обзор");
            ui.selectable_value(&mut self.view_mode, GraphViewMode::Entities, "Объекты");
            ui.selectable_value(&mut self.view_mode, GraphViewMode::Types, "Типы");
            ui.selectable_value(&mut self.view_mode, GraphViewMode::Clusters, "Кластеры");
            if self.map_domain == GraphDomain::Code {
                ui.selectable_value(&mut self.view_mode, GraphViewMode::Files, "Файлы");
            }
            ui.separator();
            if ui
                .add_enabled(!self.selection_back.is_empty(), egui::Button::new("←"))
                .on_hover_text("Назад")
                .clicked()
            {
                self.go_back();
            }
            if ui
                .add_enabled(!self.selection_forward.is_empty(), egui::Button::new("→"))
                .on_hover_text("Вперед")
                .clicked()
            {
                self.go_forward();
            }
            if ui.button("Центр").clicked() {
                self.center_selection();
            }
            if ui.button("Вписать").clicked() {
                self.fit_requested = true;
            }
            ui.separator();
            ui.label(
                RichText::new(format!("{:.0}%", self.graph_zoom * 100.0))
                    .small()
                    .color(Color32::from_rgb(151, 148, 164)),
            );
            if self.expanded_cluster.is_some()
                && ui
                    .button("Кластеры")
                    .on_hover_text("Вернуться к обзору")
                    .clicked()
            {
                self.collapse_cluster();
            }
            ui.separator();
            if let Some(status) = &self.project_status {
                ui.label(
                    RichText::new(format!(
                        "{} памяти · {} сущн. · {} кода",
                        status.total_memories,
                        self.graph.entities.len(),
                        self.code_symbols.len()
                    ))
                    .small()
                    .color(Color32::from_rgb(151, 148, 164)),
                );
            }
            if !self.last_refresh.is_empty() {
                ui.label(
                    RichText::new(&self.last_refresh)
                        .small()
                        .color(Color32::from_rgb(125, 122, 137)),
                );
            }
        });
        if previous_domain != self.map_domain || previous_view != self.view_mode {
            self.expanded_cluster = None;
            self.selected = GraphSelection::Overview;
            self.selection_back.clear();
            self.selection_forward.clear();
            self.graph_node_offsets.clear();
            self.pinned_nodes.clear();
            self.fit_requested = true;
            self.invalidate_graph_caches();
        }

        egui::CollapsingHeader::new("Фильтры")
            .id_salt("map_filters")
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Статус");
                    egui::ComboBox::from_id_salt("memory_status_filter")
                        .selected_text(self.status.label())
                        .show_ui(ui, |ui| {
                            for choice in MemoryStatusChoice::all() {
                                ui.selectable_value(&mut self.status, choice, choice.label());
                            }
                        });
                    ui.label("Тип памяти");
                    ui.add_sized(
                        [150.0, 24.0],
                        egui::TextEdit::singleline(&mut self.kind).hint_text("любой"),
                    );
                    ui.add(egui::Slider::new(&mut self.limit, 1..=500).text("Память"));
                    ui.add(egui::Slider::new(&mut self.max_visible_nodes, 20..=250).text("Узлы"));
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Узел");
                    ui.add_sized(
                        [220.0, 24.0],
                        egui::TextEdit::singleline(&mut self.node_filter)
                            .hint_text("название или тип"),
                    );
                    ui.label("Связь");
                    ui.add_sized(
                        [180.0, 24.0],
                        egui::TextEdit::singleline(&mut self.relation_filter)
                            .hint_text("uses, depends_on"),
                    );
                });
                if matches!(self.map_domain, GraphDomain::Code | GraphDomain::Bridge) {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Вид кода");
                        ui.add_sized(
                            [150.0, 24.0],
                            egui::TextEdit::singleline(&mut self.code_kind_filter)
                                .hint_text("function, struct"),
                        );
                        ui.label("Файл");
                        ui.add_sized(
                            [280.0, 24.0],
                            egui::TextEdit::singleline(&mut self.code_file_filter)
                                .hint_text("src/gui.rs"),
                        );
                    });
                }
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut self.focus_selected, "Фокус");
                    ui.checkbox(&mut self.show_facts, "Факты");
                    ui.checkbox(&mut self.show_evidence, "Основания");
                    ui.checkbox(&mut self.show_minimap, "Миникарта");
                    ui.checkbox(&mut self.show_legend, "Легенда");
                    ui.checkbox(&mut self.show_quality, "Качество");
                    if ui.button("Снять пины").clicked() {
                        self.pinned_nodes.clear();
                        self.graph_node_offsets.clear();
                        self.fit_requested = true;
                        self.save_current_project_view();
                    }
                    if ui.button("Применить").clicked() {
                        self.center_selection();
                        self.refresh();
                    }
                    if ui.button("Сбросить").clicked() {
                        self.reset_filters();
                    }
                    if ui.button("Экспорт JSON").clicked() {
                        self.export_current_map();
                    }
                });
            });

        if let Some(error) = &self.error {
            ui.add_space(4.0);
            ui.colored_label(Color32::from_rgb(232, 92, 92), format!("Ошибка: {error}"));
        }
        if self.graph_note.is_some() || self.code_note.is_some() {
            ui.horizontal_wrapped(|ui| {
                if let Some(note) = &self.graph_note {
                    ui.colored_label(Color32::from_rgb(205, 146, 71), note);
                }
                if let Some(note) = &self.code_note {
                    if self.graph_note.is_some() {
                        ui.separator();
                    }
                    ui.colored_label(Color32::from_rgb(205, 146, 71), note);
                }
            });
        }
    }

    fn graph_ui(&mut self, ui: &mut egui::Ui, model: &GraphModel) {
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new(format!(
                    "{} узлов · {} связей",
                    model.nodes.len(),
                    model.edges.len()
                ))
                .small()
                .color(Color32::from_rgb(151, 148, 164)),
            );
            ui.separator();
            ui.label(
                RichText::new(format!(
                    "{} / {}",
                    self.map_domain.label(),
                    self.view_mode.label()
                ))
                .small()
                .strong()
                .color(Color32::from_rgb(202, 191, 255)),
            );
            if self.view_mode != GraphViewMode::Overview
                && ui
                    .small_button("Обзор")
                    .on_hover_text("Вернуться к обзорной карте")
                    .clicked()
            {
                self.open_graph_view(self.map_domain, GraphViewMode::Overview);
            }
            if let Some(cluster_id) = &self.expanded_cluster {
                ui.label(
                    RichText::new(format!("кластер: {}", compact_graph_id(cluster_id)))
                        .small()
                        .color(Color32::from_rgb(151, 148, 164)),
                );
            }
            if model.fallback {
                ui.separator();
                ui.colored_label(Color32::from_rgb(205, 146, 71), "явных связей мало");
            }
        });
        ui.add_space(4.0);

        let canvas_size = Vec2::new(
            ui.available_width().max(420.0),
            ui.available_height().max(420.0),
        );
        let (rect, background_response) =
            ui.allocate_exact_size(canvas_size, Sense::click_and_drag());
        let mut next_selection = None;
        let mut node_interacted = false;
        let mut node_dragged = false;
        let mut expand_cluster = None;
        let mut collision_requested = false;

        if background_response.hovered() {
            let zoom_delta = ui.input(|input| {
                let wheel_zoom = if input.smooth_scroll_delta.y.abs() > 0.1 {
                    (input.smooth_scroll_delta.y * 0.0015).exp()
                } else {
                    1.0
                };
                input.zoom_delta() * wheel_zoom
            });
            if let Some(focus) = ui
                .input(|input| input.pointer.hover_pos())
                .filter(|pos| rect.contains(*pos))
            {
                let previous_zoom = self.graph_zoom;
                self.zoom_graph_at(rect, focus, zoom_delta);
                collision_requested |= (self.graph_zoom - previous_zoom).abs() > 0.001;
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 8.0, Color32::from_rgb(24, 24, 31));
        let clip_rect = ui.clip_rect().intersect(rect).expand(140.0);
        if model.nodes.is_empty() {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Нет записей для выбранных фильтров",
                FontId::proportional(16.0),
                Color32::from_rgb(153, 150, 166),
            );
            return;
        }

        let layout_key = layout_fingerprint(model, &self.selected, rect);
        let mut layout_rebuilt = false;
        if self
            .layout_cache
            .as_ref()
            .is_none_or(|cache| cache.key != layout_key)
        {
            self.layout_cache = Some(LayoutCache {
                key: layout_key,
                positions: layout_graph(model, rect, &self.selected),
            });
            layout_rebuilt = true;
        }
        let positions = self
            .layout_cache
            .as_ref()
            .map(|cache| cache.positions.clone())
            .unwrap_or_default();
        self.graph_node_offsets
            .retain(|node_id, _| positions.contains_key(node_id));
        if layout_rebuilt || collision_requested {
            resolve_node_offset_overlaps(
                model,
                &positions,
                &mut self.graph_node_offsets,
                self.graph_zoom,
                layout_fixed_node_id(model, &self.selected),
                &self.pinned_nodes,
            );
        }
        if self.fit_requested {
            self.fit_graph_to_view(rect, model, &positions);
            resolve_node_offset_overlaps(
                model,
                &positions,
                &mut self.graph_node_offsets,
                self.graph_zoom,
                layout_fixed_node_id(model, &self.selected),
                &self.pinned_nodes,
            );
            self.fit_requested = false;
        }

        let selected_neighbors = selected_neighbor_nodes(model, &self.selected);
        let selected_edge_ids = selected_edges(model, &self.selected);
        let overview = matches!(self.selected, GraphSelection::Overview);
        let draw_all_edge_labels = model.edges.len() <= 48 && self.graph_zoom >= 0.72;
        let parallel_edge_offsets = parallel_edge_offsets(model);

        for edge in &model.edges {
            let Some(raw_from) = transformed_graph_position(
                positions.get(&edge.from),
                self.graph_node_offsets.get(&edge.from),
                rect,
                self.graph_zoom,
                self.graph_pan,
            ) else {
                continue;
            };
            let Some(raw_to) = transformed_graph_position(
                positions.get(&edge.to),
                self.graph_node_offsets.get(&edge.to),
                rect,
                self.graph_zoom,
                self.graph_pan,
            ) else {
                continue;
            };
            let (from, to) = offset_edge_points(
                raw_from,
                raw_to,
                parallel_edge_offsets
                    .get(&edge.id)
                    .copied()
                    .unwrap_or_default(),
            );
            let selected = self.selected == GraphSelection::Edge(edge.id.clone());
            let related = !overview && selected_edge_ids.contains(&edge.id);
            let edge_rect = edge_bounds(from, to).expand(80.0);
            if !(selected || related || edge_rect.intersects(clip_rect)) {
                continue;
            }
            let weight_boost = (edge.weight.saturating_sub(1) as f32).sqrt().min(3.0);
            let base_color = edge_kind_color(edge.kind);
            let stroke = if selected {
                Stroke::new(3.0 + weight_boost, Color32::from_rgb(184, 167, 255))
            } else if related {
                Stroke::new(2.1 + weight_boost * 0.5, edge_kind_focus_color(edge.kind))
            } else {
                Stroke::new(1.0 + weight_boost * 0.35, base_color)
            };
            painter.line_segment([from, to], stroke);
            draw_edge_arrow(&painter, from, to, stroke);

            if !(selected || related || draw_all_edge_labels) {
                continue;
            }

            let mid = Pos2::new((from.x + to.x) * 0.5, (from.y + to.y) * 0.5);
            let label = truncate_text(&edge.label, 22);
            let label_rect = Rect::from_center_size(
                mid,
                Vec2::new((label.chars().count() as f32 * 6.5).max(52.0), 18.0),
            );
            let response = ui
                .interact(
                    label_rect,
                    ui.make_persistent_id(("graph_edge", &edge.id)),
                    Sense::click(),
                )
                .on_hover_cursor(egui::CursorIcon::PointingHand);
            if response.clicked() {
                node_interacted = true;
                next_selection = Some(GraphSelection::Edge(edge.id.clone()));
            }
            let label_fill = if selected || response.hovered() {
                Color32::from_rgb(57, 50, 82)
            } else {
                Color32::from_rgb(31, 30, 39)
            };
            painter.rect_filled(label_rect, 4.0, label_fill);
            painter.rect_stroke(
                label_rect,
                4.0,
                Stroke::new(0.8, edge_kind_focus_color(edge.kind)),
                egui::StrokeKind::Middle,
            );
            painter.text(
                label_rect.center(),
                Align2::CENTER_CENTER,
                label,
                FontId::proportional(10.5),
                Color32::from_rgb(210, 207, 222),
            );
        }

        for node in &model.nodes {
            let Some(pos) = transformed_graph_position(
                positions.get(&node.id),
                self.graph_node_offsets.get(&node.id),
                rect,
                self.graph_zoom,
                self.graph_pan,
            ) else {
                continue;
            };
            let selected = self.selected == GraphSelection::Node(node.id.clone());
            let related = matches!(self.selected, GraphSelection::Overview)
                || selected
                || selected_neighbors.contains(&node.id);
            let node_rect = node_rect(pos, node.kind);
            if !(selected || related || node_rect.intersects(clip_rect)) {
                continue;
            }
            let response = ui
                .interact(
                    node_rect,
                    ui.make_persistent_id(("graph_node", &node.id)),
                    Sense::click_and_drag(),
                )
                .on_hover_cursor(if node.kind == GraphNodeKind::Cluster {
                    egui::CursorIcon::ZoomIn
                } else {
                    egui::CursorIcon::PointingHand
                })
                .on_hover_text(node_hover_text(node, self.view_mode));
            if response.dragged() {
                let delta =
                    ui.input(|input| input.pointer.delta()) / self.graph_zoom.max(MIN_GRAPH_ZOOM);
                *self
                    .graph_node_offsets
                    .entry(node.id.clone())
                    .or_insert(Vec2::ZERO) += delta;
                self.pinned_nodes.insert(node.id.clone());
                node_dragged = true;
                node_interacted = true;
            } else if response.double_clicked()
                && self.view_mode == GraphViewMode::Overview
                && self.open_overview_area(&node.id)
            {
                node_interacted = true;
            } else if response.double_clicked()
                && self.view_mode == GraphViewMode::Clusters
                && node.kind == GraphNodeKind::Cluster
            {
                expand_cluster = Some(node.id.clone());
                node_interacted = true;
            } else if response.clicked() {
                next_selection = Some(GraphSelection::Node(node.id.clone()));
                node_interacted = true;
            }
            draw_graph_node(
                &painter,
                node,
                node_rect,
                selected,
                related,
                response.hovered(),
                self.pinned_nodes.contains(&node.id),
            );
        }

        if node_dragged {
            resolve_node_offset_overlaps(
                model,
                &positions,
                &mut self.graph_node_offsets,
                self.graph_zoom,
                layout_fixed_node_id(model, &self.selected),
                &self.pinned_nodes,
            );
            ui.ctx().request_repaint();
        }

        if self.show_legend {
            draw_graph_legend(&painter, rect);
        }
        if self.show_minimap
            && let Some(world_center) = draw_graph_minimap(
                ui,
                &painter,
                rect,
                model,
                &positions,
                &self.graph_node_offsets,
                &self.selected,
                self.graph_zoom,
                self.graph_pan,
            )
        {
            self.graph_pan = (rect.center() - world_center) * self.graph_zoom;
            node_interacted = true;
        }

        if background_response.dragged() && !node_dragged && !node_interacted {
            self.graph_pan += ui.input(|input| input.pointer.delta());
        }
        if background_response.double_clicked() && !node_interacted {
            self.fit_requested = true;
        } else if background_response.clicked()
            && !background_response.dragged()
            && !node_interacted
        {
            next_selection = Some(GraphSelection::Overview);
        }

        if let Some(node_id) = expand_cluster {
            self.expand_cluster(node_id);
            return;
        }

        if let Some(selection) = next_selection {
            self.select(selection);
        }
    }

    fn detail_ui(&mut self, ui: &mut egui::Ui, model: &GraphModel) {
        ScrollArea::vertical().show(ui, |ui| {
            match self.selected.clone() {
                GraphSelection::Overview => self.overview_detail_ui(ui, model),
                GraphSelection::Node(id) => self.node_detail_ui(ui, model, &id),
                GraphSelection::Edge(id) => self.edge_detail_ui(ui, model, &id),
            }
            if self.show_quality {
                ui.add_space(12.0);
                ui.separator();
                self.retrieval_quality_ui(ui, model);
                ui.add_space(12.0);
                ui.separator();
                self.graph_quality_ui(ui, model);
            }
            ui.add_space(12.0);
            ui.separator();
            egui::CollapsingHeader::new("Инструменты графа")
                .id_salt("graph_extract_tools")
                .default_open(self.extract_running || self.extract_report.is_some())
                .show(ui, |ui| {
                    self.graph_extract_ui(ui);
                });
        });
    }

    fn graph_extract_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Запрос");
            ui.add_sized(
                [250.0, 24.0],
                egui::TextEdit::singleline(&mut self.extract_query)
                    .hint_text("пусто = текущий список"),
            );
            ui.add(egui::Slider::new(&mut self.extract_limit, 1..=100).text("память"));
        });
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(!self.extract_running, egui::Button::new("Предпросмотр"))
                .clicked()
            {
                self.start_graph_extract(false);
            }
            if ui
                .add_enabled(
                    !self.extract_running,
                    egui::Button::new("Извлечь и применить"),
                )
                .clicked()
            {
                self.start_graph_extract(true);
            }
            let can_apply_report = self
                .extract_report
                .as_ref()
                .is_some_and(|report| !report.apply && !report.proposals.is_empty());
            if ui
                .add_enabled(
                    !self.extract_running && can_apply_report,
                    egui::Button::new("Применить предпросмотр"),
                )
                .clicked()
            {
                self.apply_extract_report();
            }
            if self.extract_running {
                ui.label(if self.extract_apply {
                    "ИИ извлекает и применяет..."
                } else {
                    "ИИ готовит предпросмотр..."
                });
            }
        });

        if let Some(report) = &self.extract_report {
            ui.add_space(6.0);
            egui::Grid::new("extract_report_grid")
                .num_columns(2)
                .spacing([14.0, 5.0])
                .striped(true)
                .show(ui, |ui| {
                    detail_row(ui, "память", &report.memories.to_string(), false);
                    detail_row(ui, "сущности", &report.proposed_entities.to_string(), false);
                    detail_row(ui, "факты", &report.proposed_facts.to_string(), false);
                    detail_row(ui, "связи", &report.proposed_edges.to_string(), false);
                    if report.apply {
                        detail_row(
                            ui,
                            "добавлено фактов",
                            &report.inserted_facts.to_string(),
                            false,
                        );
                        detail_row(
                            ui,
                            "добавлено связей",
                            &report.inserted_edges.to_string(),
                            false,
                        );
                    }
                });
            for proposal in report.proposals.iter().take(4) {
                ui.collapsing(truncate_text(&proposal.memory_id, 26), |ui| {
                    for entity in proposal.entities.iter().take(4) {
                        ui.label(format!("{} · {}", entity.entity_type, entity.name));
                    }
                    for edge in proposal.edges.iter().take(4) {
                        ui.label(format!(
                            "{} -{}-> {}",
                            edge.from_name,
                            display_relation(&edge.relation_type),
                            edge.to_name
                        ));
                    }
                });
            }
        }
    }

    fn apply_extract_report(&mut self) {
        let Some(report) = &self.extract_report else {
            return;
        };
        let proposals = report.proposals.clone();
        match apply_graph_extraction(&self.store, &self.project_id, proposals, true) {
            Ok(applied) => {
                self.extract_report = Some(applied);
                self.error = None;
                self.refresh();
            }
            Err(error) => {
                self.error = Some(format!("не удалось применить извлечение графа: {error}"))
            }
        }
    }

    fn overview_detail_ui(&mut self, ui: &mut egui::Ui, model: &GraphModel) {
        ui.heading("Обзор");
        ui.add_space(8.0);
        egui::Grid::new("overview_detail_grid")
            .num_columns(2)
            .spacing([14.0, 7.0])
            .striped(true)
            .show(ui, |ui| {
                detail_row(ui, "узлы", &model.nodes.len().to_string(), false);
                detail_row(ui, "связи", &model.edges.len().to_string(), false);
                match self.map_domain {
                    GraphDomain::Memory => {
                        detail_row(ui, "память", &self.memories.len().to_string(), false);
                        detail_row(
                            ui,
                            "сущности",
                            &self.graph.entities.len().to_string(),
                            false,
                        );
                        detail_row(ui, "факты", &self.graph.facts.len().to_string(), false);
                    }
                    GraphDomain::Code => {
                        detail_row(ui, "символы", &self.code_symbols.len().to_string(), false);
                        detail_row(
                            ui,
                            "связи кода",
                            &self.code_relations.len().to_string(),
                            false,
                        );
                    }
                    GraphDomain::Bridge => {
                        detail_row(ui, "память", &self.memories.len().to_string(), false);
                        detail_row(ui, "символы", &self.code_symbols.len().to_string(), false);
                        detail_row(
                            ui,
                            "сущности",
                            &self.graph.entities.len().to_string(),
                            false,
                        );
                    }
                }
            });
        ui.add_space(10.0);
        ui.heading("Открыть");
        ui.horizontal_wrapped(|ui| {
            if !self.memories.is_empty() && ui.button("Память").clicked() {
                self.open_graph_view(GraphDomain::Memory, GraphViewMode::Entities);
            }
            if !self.graph.entities.is_empty() && ui.button("Кластеры памяти").clicked()
            {
                self.open_graph_view(GraphDomain::Memory, GraphViewMode::Clusters);
            }
            if !self.code_symbols.is_empty() && ui.button("Код").clicked() {
                self.open_graph_view(GraphDomain::Code, GraphViewMode::Entities);
            }
            if !self.code_symbols.is_empty() && ui.button("Файлы").clicked() {
                self.open_graph_view(GraphDomain::Code, GraphViewMode::Files);
            }
            if !self.memories.is_empty()
                && !self.code_symbols.is_empty()
                && ui.button("Память ↔ код").clicked()
            {
                self.open_graph_view(GraphDomain::Bridge, GraphViewMode::Clusters);
            }
        });
        if model.fallback {
            ui.add_space(10.0);
            ui.colored_label(
                Color32::from_rgb(130, 88, 20),
                "Показаны слабые связи из метаданных. Явные связи появятся после извлечения сущностей, фактов и ребер.",
            );
        }
        ui.add_space(12.0);
        self.related_links(ui, model, None);
    }

    fn graph_quality_ui(&self, ui: &mut egui::Ui, model: &GraphModel) {
        let quality = analyze_graph_quality(
            model,
            &self.memories,
            &self.graph,
            &self.code_symbols,
            &self.code_relations,
            self.max_visible_nodes,
        );
        egui::CollapsingHeader::new("Качество графа")
            .id_salt("graph_quality")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("graph_quality_grid")
                    .num_columns(2)
                    .spacing([14.0, 7.0])
                    .striped(true)
                    .show(ui, |ui| {
                        detail_row(
                            ui,
                            "изолированные",
                            &quality.isolated_nodes.to_string(),
                            false,
                        );
                        detail_row(
                            ui,
                            "плотность",
                            &format!("{:.2}%", quality.density * 100.0),
                            false,
                        );
                        detail_row(ui, "явные", &quality.strong_edges.to_string(), false);
                        detail_row(ui, "слабые", &quality.weak_edges.to_string(), false);
                        detail_row(
                            ui,
                            "память без графа",
                            &quality.memories_without_graph.to_string(),
                            false,
                        );
                        if !self.code_symbols.is_empty() {
                            detail_row(
                                ui,
                                "символы без связей",
                                &quality.code_symbols_without_relations.to_string(),
                                false,
                            );
                        }
                    });
                ui.add_space(8.0);
                let color = if quality.warning_count == 0 {
                    Color32::from_rgb(109, 190, 127)
                } else if quality.warning_count <= 2 {
                    Color32::from_rgb(205, 146, 71)
                } else {
                    Color32::from_rgb(232, 92, 92)
                };
                ui.colored_label(color, quality.summary());
            });
    }

    fn retrieval_quality_ui(&self, ui: &mut egui::Ui, model: &GraphModel) {
        let quality = analyze_retrieval_quality(
            self.query.trim(),
            &self.memories,
            &self.graph,
            &self.code_symbols,
            &self.code_relations,
            model,
        );
        egui::CollapsingHeader::new("Качество поиска")
            .id_salt("retrieval_quality")
            .default_open(
                matches!(self.selected, GraphSelection::Overview)
                    || quality.query_active
                    || quality.warning_count > 0,
            )
            .show(ui, |ui| {
                let color = if quality.warning_count == 0 {
                    Color32::from_rgb(109, 190, 127)
                } else if quality.warning_count <= 2 {
                    Color32::from_rgb(205, 146, 71)
                } else {
                    Color32::from_rgb(232, 92, 92)
                };
                ui.colored_label(color, quality.summary());
                ui.add_space(8.0);
                egui::Grid::new("retrieval_quality_grid")
                    .num_columns(2)
                    .spacing([14.0, 7.0])
                    .striped(true)
                    .show(ui, |ui| {
                        detail_row(
                            ui,
                            "режим",
                            if quality.query_active {
                                "по запросу"
                            } else {
                                "проект"
                            },
                            false,
                        );
                        detail_row(
                            ui,
                            "источники",
                            &format!("{} из 5", quality.source_count),
                            false,
                        );
                        detail_row(
                            ui,
                            "покрытие памяти",
                            &format_percent(quality.memory_graph_coverage),
                            false,
                        );
                        detail_row(
                            ui,
                            "покрытие кода",
                            &format_percent(quality.code_relation_coverage),
                            false,
                        );
                        detail_row(
                            ui,
                            "узлы на карте",
                            &quality.visible_nodes.to_string(),
                            false,
                        );
                        detail_row(
                            ui,
                            "связи на карте",
                            &quality.visible_edges.to_string(),
                            false,
                        );
                    });

                ui.add_space(8.0);
                retrieval_source_meter(ui, "Память", quality.memory_hits, self.limit);
                retrieval_source_meter(ui, "Граф памяти", quality.memory_graph_signals, self.limit);
                retrieval_source_meter(ui, "Код", quality.code_symbols, self.limit);
                retrieval_source_meter(ui, "Связи кода", quality.code_relations, self.limit);
                retrieval_source_meter(ui, "Мост память-код", quality.bridge_edges, self.limit);

                let recommendations = quality.recommendations();
                if !recommendations.is_empty() {
                    ui.add_space(8.0);
                    for recommendation in recommendations {
                        ui.label(recommendation);
                    }
                }
            });
    }

    fn node_detail_ui(&mut self, ui: &mut egui::Ui, model: &GraphModel, id: &str) {
        let Some(node) = model.nodes.iter().find(|node| node.id == id).cloned() else {
            ui.label("Узел не найден");
            return;
        };

        ui.horizontal_wrapped(|ui| {
            ui.heading(&node.label);
            ui.label(node.kind.label());
            let pinned = self.pinned_nodes.contains(&node.id);
            if ui
                .button(if pinned {
                    "Открепить"
                } else {
                    "Закрепить"
                })
                .clicked()
            {
                self.toggle_pin(&node.id);
            }
        });
        ui.add_space(8.0);

        if node.id.starts_with("overview:") {
            self.overview_node_detail_ui(ui, &node);
        } else {
            match node.kind {
                GraphNodeKind::Type => self.type_node_detail(ui, &node, model),
                GraphNodeKind::Cluster => self.cluster_node_detail(ui, &node, model),
                GraphNodeKind::CodeSymbol => self.code_symbol_node_detail(ui, &node),
                GraphNodeKind::CodeFile => self.code_file_node_detail(ui, &node, model),
                GraphNodeKind::Memory => self.memory_node_detail(ui, &node),
                GraphNodeKind::Entity => self.entity_node_detail(ui, &node),
                GraphNodeKind::Fact => self.fact_node_detail(ui, &node),
            }
        }

        ui.add_space(12.0);
        self.explain_paths_ui(ui, model, &node.id);
        ui.add_space(12.0);
        self.related_links(ui, model, Some(&node.id));
    }

    fn edge_detail_ui(&mut self, ui: &mut egui::Ui, model: &GraphModel, id: &str) {
        let Some(edge) = model.edges.iter().find(|edge| edge.id == id).cloned() else {
            ui.label("Связь не найдена");
            return;
        };
        ui.heading(&edge.label);
        ui.add_space(8.0);
        egui::Grid::new("edge_detail_grid")
            .num_columns(2)
            .spacing([14.0, 7.0])
            .striped(true)
            .show(ui, |ui| {
                detail_row(ui, "от", node_label(model, &edge.from), false);
                detail_row(ui, "к", node_label(model, &edge.to), false);
                detail_row(ui, "тип", edge.kind.label(), false);
                if edge.weight > 1 {
                    detail_row(ui, "количество", &edge.weight.to_string(), false);
                }
            });
        ui.add_space(10.0);
        ui.horizontal_wrapped(|ui| {
            if ui.button("К началу").clicked() {
                self.select(GraphSelection::Node(edge.from.clone()));
            }
            if ui.button("К концу").clicked() {
                self.select(GraphSelection::Node(edge.to.clone()));
            }
        });
        ui.add_space(12.0);
        self.edge_explanation_ui(ui, model, &edge);
    }

    fn overview_node_detail_ui(&mut self, ui: &mut egui::Ui, node: &GraphNode) {
        egui::Grid::new("overview_node_detail_grid")
            .num_columns(2)
            .spacing([14.0, 7.0])
            .striped(true)
            .show(ui, |ui| {
                detail_row(ui, "область", &node.label, false);
                detail_row(ui, "детали", &node.detail, false);
            });

        if let Some(target) = overview_navigation_target(&node.id) {
            ui.add_space(10.0);
            if ui.button(target.action_label).clicked() {
                if target.show_facts {
                    self.show_facts = true;
                }
                self.open_graph_view(target.domain, target.view_mode);
            }
        }
    }

    fn type_node_detail(&mut self, ui: &mut egui::Ui, node: &GraphNode, model: &GraphModel) {
        let Some(entity_type) = node.id.strip_prefix("type:") else {
            return;
        };
        let entity_count = self
            .graph
            .entities
            .iter()
            .filter(|entity| entity.entity_type == entity_type)
            .count();
        let relation_count = model
            .edges
            .iter()
            .filter(|edge| edge.from == node.id || edge.to == node.id)
            .map(|edge| edge.weight.max(1))
            .sum::<usize>();
        egui::Grid::new("type_detail_grid")
            .num_columns(2)
            .spacing([14.0, 7.0])
            .striped(true)
            .show(ui, |ui| {
                detail_row(ui, "тип", entity_type, false);
                detail_row(ui, "сущности", &entity_count.to_string(), false);
                detail_row(ui, "связи", &relation_count.to_string(), false);
            });
    }

    fn cluster_node_detail(&mut self, ui: &mut egui::Ui, node: &GraphNode, model: &GraphModel) {
        let relation_count = model
            .edges
            .iter()
            .filter(|edge| edge.from == node.id || edge.to == node.id)
            .map(|edge| edge.weight.max(1))
            .sum::<usize>();
        egui::Grid::new("cluster_detail_grid")
            .num_columns(2)
            .spacing([14.0, 7.0])
            .striped(true)
            .show(ui, |ui| {
                detail_row(ui, "кластер", &node.label, false);
                detail_row(ui, "детали", &node.detail, false);
                detail_row(ui, "связи", &relation_count.to_string(), false);
            });
        if self.view_mode == GraphViewMode::Clusters && ui.button("Открыть кластер").clicked()
        {
            self.expand_cluster(node.id.clone());
        }
    }

    fn code_symbol_node_detail(&mut self, ui: &mut egui::Ui, node: &GraphNode) {
        let Some(symbol_id) = node.id.strip_prefix("code:") else {
            return;
        };
        let Some(symbol) = self
            .code_symbols
            .iter()
            .find(|symbol| symbol.id == symbol_id)
        else {
            ui.label(&node.detail);
            return;
        };
        egui::Grid::new("code_symbol_detail_grid")
            .num_columns(2)
            .spacing([14.0, 7.0])
            .striped(true)
            .show(ui, |ui| {
                detail_row(ui, "id", &symbol.id, true);
                detail_row(ui, "вид", &symbol.kind, false);
                detail_row(ui, "файл", &symbol.file_path, false);
                detail_row(
                    ui,
                    "строки",
                    &format!("{}-{}", symbol.start_line, symbol.end_line),
                    false,
                );
                if let Some(parent_id) = &symbol.parent_id {
                    detail_row(ui, "родитель", parent_id, true);
                }
            });
        ui.add_space(12.0);
        ui.heading("Сигнатура");
        ui.separator();
        ui.label(RichText::new(&symbol.signature).monospace());
        ui.add_space(10.0);
        if ui.button("Открыть файл").clicked() {
            let file_path = symbol.file_path.clone();
            self.open_code_file(&file_path);
        }
    }

    fn code_file_node_detail(&mut self, ui: &mut egui::Ui, node: &GraphNode, model: &GraphModel) {
        let Some(file_path) = node.id.strip_prefix("code-file:") else {
            return;
        };
        let symbol_count = self
            .code_symbols
            .iter()
            .filter(|symbol| symbol.file_path == file_path)
            .count();
        let relation_count = model
            .edges
            .iter()
            .filter(|edge| edge.from == node.id || edge.to == node.id)
            .map(|edge| edge.weight.max(1))
            .sum::<usize>();
        egui::Grid::new("code_file_detail_grid")
            .num_columns(2)
            .spacing([14.0, 7.0])
            .striped(true)
            .show(ui, |ui| {
                detail_row(ui, "файл", file_path, false);
                detail_row(ui, "символы", &symbol_count.to_string(), false);
                detail_row(ui, "связи", &relation_count.to_string(), false);
            });
        ui.add_space(10.0);
        if ui.button("Открыть файл").clicked() {
            self.open_code_file(file_path);
        }
    }

    fn memory_node_detail(&mut self, ui: &mut egui::Ui, node: &GraphNode) {
        let Some(memory_id) = node.id.strip_prefix("memory:") else {
            return;
        };
        let Some(memory) = self.memories.iter().find(|memory| memory.id == memory_id) else {
            ui.label(RichText::new(memory_id).monospace());
            return;
        };
        ui.horizontal_wrapped(|ui| {
            status_badge(ui, &memory.status);
            ui.label(RichText::new(&memory.kind).strong());
        });
        ui.add_space(8.0);
        egui::Grid::new("memory_detail_grid")
            .num_columns(2)
            .spacing([14.0, 7.0])
            .striped(true)
            .show(ui, |ui| {
                detail_row(ui, "id", &memory.id, true);
                detail_row(ui, "создано", &memory.created_at, false);
                detail_row(ui, "обновлено", &memory.updated_at, false);
                detail_row(ui, "важность", &format!("{:.2}", memory.importance), false);
                detail_row(
                    ui,
                    "уверенность",
                    &format!("{:.2}", memory.confidence),
                    false,
                );
                if !memory.tags.is_empty() {
                    detail_row(ui, "теги", &memory.tags.join(", "), false);
                }
                if let Some(source) = &memory.source {
                    detail_row(ui, "источник", source, false);
                }
            });
        ui.add_space(12.0);
        ui.heading("Текст");
        ui.separator();
        ui.add(egui::Label::new(memory.body.as_str()).wrap());
    }

    fn entity_node_detail(&mut self, ui: &mut egui::Ui, node: &GraphNode) {
        let Some(entity_id) = node.id.strip_prefix("entity:") else {
            return;
        };
        let Some(entity) = self
            .graph
            .entities
            .iter()
            .find(|entity| entity.id == entity_id)
        else {
            ui.label(&node.detail);
            return;
        };
        egui::Grid::new("entity_detail_grid")
            .num_columns(2)
            .spacing([14.0, 7.0])
            .striped(true)
            .show(ui, |ui| {
                detail_row(ui, "id", &entity.id, true);
                detail_row(ui, "тип", &entity.entity_type, false);
                if !entity.aliases.is_empty() {
                    detail_row(ui, "синонимы", &entity.aliases.join(", "), false);
                }
                if let Some(description) = &entity.description {
                    detail_row(ui, "описание", description, false);
                }
                detail_row(ui, "обновлено", &entity.updated_at, false);
            });
    }

    fn fact_node_detail(&mut self, ui: &mut egui::Ui, node: &GraphNode) {
        let Some(fact_id) = node.id.strip_prefix("fact:") else {
            return;
        };
        let Some(fact) = self.graph.facts.iter().find(|fact| fact.id == fact_id) else {
            ui.label(&node.detail);
            return;
        };
        egui::Grid::new("fact_detail_grid")
            .num_columns(2)
            .spacing([14.0, 7.0])
            .striped(true)
            .show(ui, |ui| {
                detail_row(ui, "id", &fact.id, true);
                detail_row(ui, "предикат", &fact.predicate, false);
                detail_row(ui, "значение", &fact.value, false);
                detail_row(ui, "уверенность", &format!("{:.2}", fact.confidence), false);
                detail_row(ui, "наблюдалось", &fact.observed_at, false);
            });
    }

    fn explain_paths_ui(&mut self, ui: &mut egui::Ui, model: &GraphModel, node_id: &str) {
        let paths = explainable_paths(model, node_id, 6);
        if paths.is_empty() {
            return;
        }

        ui.heading("Пути");
        ui.separator();
        for path in paths {
            let target_id = path
                .node_ids
                .last()
                .cloned()
                .unwrap_or_else(|| node_id.to_string());
            let text = format_explainable_path(model, &path);
            if ui
                .add(egui::Label::new(text).wrap().sense(Sense::click()))
                .on_hover_text("Открыть конечный узел пути")
                .clicked()
            {
                self.select(GraphSelection::Node(target_id));
            }
        }
    }

    fn edge_explanation_ui(&mut self, ui: &mut egui::Ui, model: &GraphModel, edge: &GraphEdge) {
        let path = ExplainablePath {
            node_ids: vec![edge.from.clone(), edge.to.clone()],
            edge_ids: vec![edge.id.clone()],
            score: edge.weight.max(1),
        };
        ui.heading("Путь");
        ui.separator();
        if ui
            .add(
                egui::Label::new(format_explainable_path(model, &path))
                    .wrap()
                    .sense(Sense::click()),
            )
            .on_hover_text("Открыть конечный узел")
            .clicked()
        {
            self.select(GraphSelection::Node(edge.to.clone()));
        }
    }

    fn related_links(&mut self, ui: &mut egui::Ui, model: &GraphModel, node_id: Option<&str>) {
        let edges = model
            .edges
            .iter()
            .filter(|edge| node_id.is_none_or(|id| edge.from == id || edge.to == id))
            .cloned()
            .collect::<Vec<_>>();
        if edges.is_empty() {
            return;
        }

        ui.heading("Связи");
        ui.separator();
        for edge in edges.into_iter().take(40) {
            let text = format!(
                "{}  {} -> {}",
                edge.label,
                node_label(model, &edge.from),
                node_label(model, &edge.to)
            );
            let selected = self.selected == GraphSelection::Edge(edge.id.clone());
            if ui.selectable_label(selected, text).clicked() {
                self.select(GraphSelection::Edge(edge.id));
            }
        }
    }

    fn expanded_cluster_model(&self) -> Option<GraphModel> {
        let cluster_id = self.expanded_cluster.as_deref()?;
        match self.map_domain {
            GraphDomain::Memory => {
                build_memory_cluster_drilldown_model(&self.memories, &self.graph, cluster_id)
            }
            GraphDomain::Code => build_code_cluster_drilldown_model(
                &self.code_symbols,
                &self.code_relations,
                cluster_id,
            ),
            GraphDomain::Bridge => build_bridge_cluster_drilldown_model(
                &self.memories,
                &self.graph,
                &self.code_symbols,
                cluster_id,
            ),
        }
    }

    fn relationship_model(&self) -> GraphModel {
        let model = match (self.map_domain, self.view_mode) {
            (_, GraphViewMode::Overview) => build_project_overview_model(
                &self.memories,
                &self.graph,
                &self.code_symbols,
                &self.code_relations,
            ),
            (GraphDomain::Memory, GraphViewMode::Entities) => {
                build_relationship_model(&self.memories, &self.graph)
            }
            (GraphDomain::Memory, GraphViewMode::Types) => build_type_model(&self.graph),
            (GraphDomain::Memory, GraphViewMode::Clusters) => self
                .expanded_cluster_model()
                .unwrap_or_else(|| build_memory_cluster_model(&self.graph)),
            (GraphDomain::Code, GraphViewMode::Entities) => {
                build_code_symbol_model(&self.code_symbols, &self.code_relations)
            }
            (GraphDomain::Code, GraphViewMode::Types) => {
                build_code_kind_model(&self.code_symbols, &self.code_relations)
            }
            (GraphDomain::Code, GraphViewMode::Clusters) => {
                self.expanded_cluster_model().unwrap_or_else(|| {
                    build_code_directory_model(&self.code_symbols, &self.code_relations)
                })
            }
            (GraphDomain::Code, GraphViewMode::Files) => {
                build_code_file_model(&self.code_symbols, &self.code_relations)
            }
            (GraphDomain::Bridge, GraphViewMode::Clusters) => {
                self.expanded_cluster_model().unwrap_or_else(|| {
                    build_bridge_cluster_model(&self.memories, &self.graph, &self.code_symbols)
                })
            }
            (GraphDomain::Bridge, _) => {
                build_bridge_model(&self.memories, &self.graph, &self.code_symbols)
            }
            (GraphDomain::Memory, GraphViewMode::Files) => build_type_model(&self.graph),
        };
        filter_relationship_model(
            model,
            RelationshipFilterOptions {
                selection: &self.selected,
                show_facts: self.show_facts,
                show_evidence: self.show_evidence,
                focus_selected: self.focus_selected,
                max_visible_nodes: self.max_visible_nodes,
                node_filter: &self.node_filter,
                relation_filter: &self.relation_filter,
            },
        )
    }
}

impl eframe::App for MemoryViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(obsidian_visuals());
        self.poll_graph_extract();
        egui::TopBottomPanel::top("memory_viewer_header").show(ctx, |ui| {
            self.header_ui(ui);
        });
        let model = self.validated_model();
        egui::SidePanel::left("memory_viewer_projects")
            .resizable(true)
            .default_width(215.0)
            .min_width(165.0)
            .show(ctx, |ui| {
                self.project_sidebar_ui(ui);
            });
        egui::SidePanel::right("memory_viewer_detail")
            .resizable(true)
            .default_width(360.0)
            .min_width(280.0)
            .show(ctx, |ui| {
                self.detail_ui(ui, &model);
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.graph_ui(ui, &model);
        });
    }
}

impl Drop for MemoryViewerApp {
    fn drop(&mut self) {
        self.save_settings();
    }
}

struct GraphExtractJob {
    database_marker: PathBuf,
    ollama_base_url: String,
    ollama_llm_model: String,
    project_id: String,
    query: String,
    limit: usize,
    status: MemoryStatusChoice,
    kind: Option<String>,
    apply: bool,
}

fn run_graph_extract_job(job: GraphExtractJob) -> Result<GraphExtractionReport> {
    let store = Store::open(&job.database_marker)?;
    let memories = if job.query.trim().is_empty() {
        store.list(
            &job.project_id,
            ListOptions {
                limit: job.limit,
                offset: 0,
                status: job.status.filter(),
                kind: job.kind,
                memory_tier: None,
            },
        )?
    } else {
        store.search(
            &job.project_id,
            SearchOptions {
                query: job.query,
                limit: job.limit,
                status: job.status.filter(),
                kind: job.kind,
                memory_tier: None,
            },
        )?
    };
    let ollama = OllamaClient::new(job.ollama_base_url, job.ollama_llm_model);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let proposals = runtime.block_on(extract_memory_graph(&ollama, &job.project_id, &memories))?;
    apply_graph_extraction(&store, &job.project_id, proposals, job.apply)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GraphSelection {
    Overview,
    Node(String),
    Edge(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum GraphDomain {
    Memory,
    Code,
    Bridge,
}

impl GraphDomain {
    fn label(self) -> &'static str {
        match self {
            Self::Memory => "Память",
            Self::Code => "Код",
            Self::Bridge => "Связи",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum GraphViewMode {
    Overview,
    Entities,
    Types,
    Clusters,
    Files,
}

impl GraphViewMode {
    fn label(self) -> &'static str {
        match self {
            Self::Overview => "Обзор",
            Self::Entities => "Объекты",
            Self::Types => "Типы",
            Self::Clusters => "Кластеры",
            Self::Files => "Файлы",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OverviewNavigationTarget {
    domain: GraphDomain,
    view_mode: GraphViewMode,
    action_label: &'static str,
    show_facts: bool,
}

fn overview_navigation_target(node_id: &str) -> Option<OverviewNavigationTarget> {
    let target = match node_id {
        "overview:project" => OverviewNavigationTarget {
            domain: GraphDomain::Memory,
            view_mode: GraphViewMode::Overview,
            action_label: "К обзору проекта",
            show_facts: false,
        },
        "overview:memory" => OverviewNavigationTarget {
            domain: GraphDomain::Memory,
            view_mode: GraphViewMode::Entities,
            action_label: "Открыть память",
            show_facts: false,
        },
        "overview:entities" => OverviewNavigationTarget {
            domain: GraphDomain::Memory,
            view_mode: GraphViewMode::Entities,
            action_label: "Открыть сущности",
            show_facts: false,
        },
        "overview:facts" => OverviewNavigationTarget {
            domain: GraphDomain::Memory,
            view_mode: GraphViewMode::Entities,
            action_label: "Открыть факты",
            show_facts: true,
        },
        "overview:relations" => OverviewNavigationTarget {
            domain: GraphDomain::Memory,
            view_mode: GraphViewMode::Entities,
            action_label: "Открыть связи памяти",
            show_facts: false,
        },
        "overview:memory-clusters" => OverviewNavigationTarget {
            domain: GraphDomain::Memory,
            view_mode: GraphViewMode::Clusters,
            action_label: "Открыть области памяти",
            show_facts: false,
        },
        "overview:code" => OverviewNavigationTarget {
            domain: GraphDomain::Code,
            view_mode: GraphViewMode::Entities,
            action_label: "Открыть код",
            show_facts: false,
        },
        "overview:files" => OverviewNavigationTarget {
            domain: GraphDomain::Code,
            view_mode: GraphViewMode::Files,
            action_label: "Открыть файлы",
            show_facts: false,
        },
        "overview:code-clusters" => OverviewNavigationTarget {
            domain: GraphDomain::Code,
            view_mode: GraphViewMode::Clusters,
            action_label: "Открыть области кода",
            show_facts: false,
        },
        _ => return None,
    };
    Some(target)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryStatusChoice {
    Any,
    Pending,
    Active,
    Superseded,
    Archived,
}

impl MemoryStatusChoice {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "any" => Ok(Self::Any),
            "pending" => Ok(Self::Pending),
            "active" => Ok(Self::Active),
            "superseded" => Ok(Self::Superseded),
            "archived" => Ok(Self::Archived),
            other => {
                bail!(
                    "invalid memory status `{other}`; use any, pending, active, superseded, or archived"
                )
            }
        }
    }

    fn all() -> [Self; 5] {
        [
            Self::Any,
            Self::Pending,
            Self::Active,
            Self::Superseded,
            Self::Archived,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            Self::Any => "все",
            Self::Pending => "на проверке",
            Self::Active => "активные",
            Self::Superseded => "замененные",
            Self::Archived => "архив",
        }
    }

    fn filter(self) -> StatusFilter {
        match self {
            Self::Any => StatusFilter::Any,
            Self::Pending => StatusFilter::One(MemoryStatus::Pending),
            Self::Active => StatusFilter::One(MemoryStatus::Active),
            Self::Superseded => StatusFilter::One(MemoryStatus::Superseded),
            Self::Archived => StatusFilter::One(MemoryStatus::Archived),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
enum GraphNodeKind {
    Cluster,
    Type,
    CodeSymbol,
    CodeFile,
    Entity,
    Memory,
    Fact,
}

impl GraphNodeKind {
    fn label(self) -> &'static str {
        match self {
            Self::Cluster => "кластер",
            Self::Type => "тип",
            Self::CodeSymbol => "символ",
            Self::CodeFile => "файл",
            Self::Entity => "сущность",
            Self::Memory => "память",
            Self::Fact => "факт",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum GraphEdgeKind {
    Aggregate,
    CodeRelation,
    Explicit,
    Fact,
    Evidence,
    Derived,
}

impl GraphEdgeKind {
    fn label(self) -> &'static str {
        match self {
            Self::Aggregate => "сводная связь",
            Self::CodeRelation => "связь кода",
            Self::Explicit => "явная связь",
            Self::Fact => "факт",
            Self::Evidence => "основание",
            Self::Derived => "метаданные",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct GraphNode {
    id: String,
    label: String,
    detail: String,
    kind: GraphNodeKind,
}

#[derive(Debug, Clone, Serialize)]
struct GraphEdge {
    id: String,
    from: String,
    to: String,
    label: String,
    kind: GraphEdgeKind,
    weight: usize,
}

#[derive(Debug, Clone, Serialize)]
struct GraphModel {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    fallback: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExplainablePath {
    node_ids: Vec<String>,
    edge_ids: Vec<String>,
    score: usize,
}

struct CodeTarget {
    id: String,
    label: String,
    detail: String,
    kind: GraphNodeKind,
    file_path: Option<String>,
}

struct CodeTargetLookupEntry {
    file_path_lower: String,
    file_name_lower: String,
    file_id: String,
    file_label: String,
    symbol_name_lower: String,
    symbol_id: String,
    symbol_label: String,
    symbol_detail: String,
    symbol_file_path: String,
    symbol_kind: GraphNodeKind,
}

struct CodeTargetIndex {
    entries: Vec<CodeTargetLookupEntry>,
}

impl CodeTargetIndex {
    fn new(symbols: &[CodeSymbol]) -> Self {
        let entries = symbols
            .iter()
            .map(|symbol| {
                let file_label = file_label(&symbol.file_path);
                CodeTargetLookupEntry {
                    file_path_lower: symbol.file_path.to_ascii_lowercase(),
                    file_name_lower: file_label.to_ascii_lowercase(),
                    file_id: code_file_node_id(&symbol.file_path),
                    file_label,
                    symbol_name_lower: symbol.name.to_ascii_lowercase(),
                    symbol_id: code_node_id(&symbol.id),
                    symbol_label: symbol.name.clone(),
                    symbol_detail: format!(
                        "{} · {}:{}",
                        symbol.kind, symbol.file_path, symbol.start_line
                    ),
                    symbol_file_path: symbol.file_path.clone(),
                    symbol_kind: GraphNodeKind::CodeSymbol,
                }
            })
            .collect();
        Self { entries }
    }

    fn targets_for_text(&self, text: &str, limit: usize) -> Vec<CodeTarget> {
        let mut targets = Vec::<CodeTarget>::new();
        let mut seen = HashSet::<String>::new();
        for entry in &self.entries {
            if (text.contains(&entry.file_path_lower)
                || (entry.file_name_lower.contains('.') && text.contains(&entry.file_name_lower)))
                && seen.insert(entry.file_id.clone())
            {
                targets.push(CodeTarget {
                    id: entry.file_id.clone(),
                    label: entry.file_label.clone(),
                    detail: entry.symbol_file_path.clone(),
                    kind: GraphNodeKind::CodeFile,
                    file_path: Some(entry.symbol_file_path.clone()),
                });
            }

            if entry.symbol_name_lower.chars().count() >= 6
                && text.contains(&entry.symbol_name_lower)
                && seen.insert(entry.symbol_id.clone())
            {
                targets.push(CodeTarget {
                    id: entry.symbol_id.clone(),
                    label: entry.symbol_label.clone(),
                    detail: entry.symbol_detail.clone(),
                    kind: entry.symbol_kind,
                    file_path: Some(entry.symbol_file_path.clone()),
                });
            }

            if targets.len() >= limit {
                break;
            }
        }
        targets
    }
}

struct ModelCache {
    key: u64,
    model: GraphModel,
}

struct LayoutCache {
    key: u64,
    positions: BTreeMap<String, Pos2>,
}

impl GraphModel {
    fn contains_selection(&self, selection: &GraphSelection) -> bool {
        match selection {
            GraphSelection::Overview => true,
            GraphSelection::Node(id) => self.nodes.iter().any(|node| &node.id == id),
            GraphSelection::Edge(id) => self.edges.iter().any(|edge| &edge.id == id),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GuiSettings {
    node_filter: String,
    relation_filter: String,
    code_kind_filter: String,
    code_file_filter: String,
    map_domain: GraphDomain,
    view_mode: GraphViewMode,
    show_facts: bool,
    show_evidence: bool,
    #[serde(default = "default_true")]
    show_minimap: bool,
    #[serde(default = "default_true")]
    show_legend: bool,
    #[serde(default = "default_true")]
    show_quality: bool,
    focus_selected: bool,
    max_visible_nodes: usize,
    #[serde(default)]
    project_views: BTreeMap<String, SavedProjectView>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct SavedVec2 {
    x: f32,
    y: f32,
}

impl SavedVec2 {
    fn from_vec2(value: Vec2) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }

    fn to_vec2(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedProjectView {
    #[serde(default)]
    pan: SavedVec2,
    #[serde(default = "default_graph_zoom")]
    zoom: f32,
    #[serde(default)]
    node_offsets: BTreeMap<String, SavedVec2>,
    #[serde(default)]
    pinned_nodes: Vec<String>,
    #[serde(default = "default_graph_domain")]
    map_domain: GraphDomain,
    #[serde(default = "default_graph_view_mode")]
    view_mode: GraphViewMode,
    #[serde(default)]
    show_facts: bool,
    #[serde(default)]
    show_evidence: bool,
    #[serde(default = "default_true")]
    focus_selected: bool,
    #[serde(default = "default_max_visible_nodes")]
    max_visible_nodes: usize,
    #[serde(default)]
    expanded_cluster: Option<String>,
}

impl SavedProjectView {
    #[allow(clippy::too_many_arguments)]
    fn from_app(
        pan: Vec2,
        zoom: f32,
        node_offsets: &BTreeMap<String, Vec2>,
        pinned_nodes: &BTreeSet<String>,
        map_domain: GraphDomain,
        view_mode: GraphViewMode,
        show_facts: bool,
        show_evidence: bool,
        focus_selected: bool,
        max_visible_nodes: usize,
        expanded_cluster: Option<String>,
    ) -> Self {
        Self {
            pan: SavedVec2::from_vec2(pan),
            zoom,
            node_offsets: node_offsets
                .iter()
                .map(|(id, offset)| (id.clone(), SavedVec2::from_vec2(*offset)))
                .collect(),
            pinned_nodes: pinned_nodes.iter().cloned().collect(),
            map_domain,
            view_mode,
            show_facts,
            show_evidence,
            focus_selected,
            max_visible_nodes,
            expanded_cluster,
        }
    }
}

fn default_graph_zoom() -> f32 {
    1.0
}

fn default_graph_domain() -> GraphDomain {
    GraphDomain::Memory
}

fn default_graph_view_mode() -> GraphViewMode {
    GraphViewMode::Overview
}

fn default_max_visible_nodes() -> usize {
    80
}

#[derive(Debug, Clone)]
struct GraphQuality {
    isolated_nodes: usize,
    density: f32,
    strong_edges: usize,
    weak_edges: usize,
    memories_without_graph: usize,
    code_symbols_without_relations: usize,
    warning_count: usize,
}

impl GraphQuality {
    fn summary(&self) -> String {
        if self.warning_count == 0 {
            "граф выглядит связным".to_string()
        } else {
            format!("предупреждений: {}", self.warning_count)
        }
    }
}

#[derive(Debug, Clone)]
struct RetrievalQuality {
    query_active: bool,
    memory_hits: usize,
    memory_graph_signals: usize,
    code_symbols: usize,
    code_relations: usize,
    bridge_edges: usize,
    source_count: usize,
    memory_graph_coverage: f32,
    code_relation_coverage: f32,
    visible_nodes: usize,
    visible_edges: usize,
    warning_count: usize,
}

impl RetrievalQuality {
    fn summary(&self) -> String {
        if self.source_count == 0 {
            "нет данных для текущей карты".to_string()
        } else if self.warning_count == 0 {
            "источники поиска сбалансированы".to_string()
        } else {
            format!("слабых мест: {}", self.warning_count)
        }
    }

    fn recommendations(&self) -> Vec<&'static str> {
        let mut recommendations = Vec::new();
        if self.query_active && self.source_count <= 1 {
            recommendations.push("результат держится на одном источнике");
        }
        if self.memory_hits > 0 && self.memory_graph_coverage < 0.25 {
            recommendations.push("мало графовых оснований для найденной памяти");
        }
        if self.code_symbols > 0 && self.code_relation_coverage < 0.25 {
            recommendations.push("мало связей между найденными символами");
        }
        if self.memory_hits > 0 && self.code_symbols > 0 && self.bridge_edges == 0 {
            recommendations.push("нет мостов память-код");
        }
        recommendations
    }
}

fn analyze_retrieval_quality(
    query: &str,
    memories: &[Memory],
    graph: &MemoryGraph,
    code_symbols: &[CodeSymbol],
    code_relations: &[CodeRelation],
    model: &GraphModel,
) -> RetrievalQuality {
    let graph_memory_ids = graph
        .facts
        .iter()
        .filter_map(|fact| fact.memory_id.as_deref())
        .chain(
            graph
                .edges
                .iter()
                .filter_map(|edge| edge.memory_id.as_deref()),
        )
        .collect::<BTreeSet<_>>();
    let memory_graph_hits = memories
        .iter()
        .filter(|memory| graph_memory_ids.contains(memory.id.as_str()))
        .count();
    let memory_graph_coverage = ratio(memory_graph_hits, memories.len());
    let code_symbols_with_relations = code_relations
        .iter()
        .flat_map(|relation| {
            [
                relation.from_symbol_id.as_deref(),
                relation.target_symbol_id.as_deref(),
            ]
            .into_iter()
            .flatten()
        })
        .collect::<BTreeSet<_>>();
    let code_relation_hits = code_symbols
        .iter()
        .filter(|symbol| code_symbols_with_relations.contains(symbol.id.as_str()))
        .count();
    let code_relation_coverage = ratio(code_relation_hits, code_symbols.len());
    let memory_graph_signals = graph.entities.len() + graph.facts.len() + graph.edges.len();
    let bridge_edges = count_bridge_matches(memories, graph, code_symbols);
    let source_count = usize::from(!memories.is_empty())
        + usize::from(memory_graph_signals > 0)
        + usize::from(!code_symbols.is_empty())
        + usize::from(!code_relations.is_empty())
        + usize::from(bridge_edges > 0);
    let warning_count = usize::from(!query.trim().is_empty() && source_count <= 1)
        + usize::from(!memories.is_empty() && memory_graph_coverage < 0.25)
        + usize::from(!code_symbols.is_empty() && code_relation_coverage < 0.25)
        + usize::from(!memories.is_empty() && !code_symbols.is_empty() && bridge_edges == 0);

    RetrievalQuality {
        query_active: !query.trim().is_empty(),
        memory_hits: memories.len(),
        memory_graph_signals,
        code_symbols: code_symbols.len(),
        code_relations: code_relations.len(),
        bridge_edges,
        source_count,
        memory_graph_coverage,
        code_relation_coverage,
        visible_nodes: model.nodes.len(),
        visible_edges: model.edges.len(),
        warning_count,
    }
}

fn count_bridge_matches(memories: &[Memory], graph: &MemoryGraph, symbols: &[CodeSymbol]) -> usize {
    if symbols.is_empty() {
        return 0;
    }

    let target_index = CodeTargetIndex::new(symbols);
    let mut matches = BTreeSet::<String>::new();
    for memory in memories {
        for target in target_index.targets_for_text(&memory_search_text(memory), 4) {
            matches.insert(format!("memory:{}->{}", memory.id, target.id));
        }
    }
    for entity in &graph.entities {
        for target in target_index.targets_for_text(&entity_search_text(entity), 4) {
            matches.insert(format!("entity:{}->{}", entity.id, target.id));
        }
    }
    matches.len()
}

fn ratio(part: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        (part as f32 / total as f32).clamp(0.0, 1.0)
    }
}

fn analyze_graph_quality(
    model: &GraphModel,
    memories: &[Memory],
    graph: &MemoryGraph,
    code_symbols: &[CodeSymbol],
    code_relations: &[CodeRelation],
    max_visible_nodes: usize,
) -> GraphQuality {
    let mut connected = BTreeSet::new();
    let strong_edges = model
        .edges
        .iter()
        .filter(|edge| !matches!(edge.kind, GraphEdgeKind::Derived | GraphEdgeKind::Aggregate))
        .count();
    let weak_edges = model.edges.len().saturating_sub(strong_edges);
    for edge in &model.edges {
        connected.insert(edge.from.clone());
        connected.insert(edge.to.clone());
    }
    let isolated_nodes = model
        .nodes
        .iter()
        .filter(|node| !connected.contains(&node.id))
        .count();
    let possible_edges = model
        .nodes
        .len()
        .saturating_mul(model.nodes.len().saturating_sub(1));
    let density = if possible_edges == 0 {
        0.0
    } else {
        model.edges.len() as f32 / possible_edges as f32
    };
    let graph_memory_ids = graph
        .facts
        .iter()
        .filter_map(|fact| fact.memory_id.as_deref())
        .chain(
            graph
                .edges
                .iter()
                .filter_map(|edge| edge.memory_id.as_deref()),
        )
        .collect::<BTreeSet<_>>();
    let memories_without_graph = memories
        .iter()
        .filter(|memory| !graph_memory_ids.contains(memory.id.as_str()))
        .count();
    let code_symbols_with_relations = code_relations
        .iter()
        .flat_map(|relation| {
            [
                relation.from_symbol_id.as_deref(),
                relation.target_symbol_id.as_deref(),
            ]
            .into_iter()
            .flatten()
        })
        .collect::<BTreeSet<_>>();
    let code_symbols_without_relations = code_symbols
        .iter()
        .filter(|symbol| !code_symbols_with_relations.contains(symbol.id.as_str()))
        .count();
    let warning_count = usize::from(isolated_nodes > max_visible_nodes / 2)
        + usize::from(
            memories_without_graph > memories.len().saturating_div(2) && !memories.is_empty(),
        )
        + usize::from(
            code_symbols_without_relations > code_symbols.len().saturating_div(2)
                && !code_symbols.is_empty(),
        );

    GraphQuality {
        isolated_nodes,
        density,
        strong_edges,
        weak_edges,
        memories_without_graph,
        code_symbols_without_relations,
        warning_count,
    }
}

fn build_project_overview_model(
    memories: &[Memory],
    graph: &MemoryGraph,
    code_symbols: &[CodeSymbol],
    code_relations: &[CodeRelation],
) -> GraphModel {
    if memories.is_empty()
        && graph.entities.is_empty()
        && graph.facts.is_empty()
        && graph.edges.is_empty()
        && code_symbols.is_empty()
    {
        return GraphModel {
            nodes: Vec::new(),
            edges: Vec::new(),
            fallback: false,
        };
    }

    let mut nodes = Vec::<GraphNode>::new();
    let mut present = BTreeSet::<String>::new();
    let mut push_node = |id: &str, label: &str, detail: String, kind: GraphNodeKind| {
        present.insert(id.to_string());
        nodes.push(GraphNode {
            id: id.to_string(),
            label: label.to_string(),
            detail,
            kind,
        });
    };

    push_node(
        "overview:project",
        "Проект",
        format!(
            "{} памяти · {} сущн. · {} симв.",
            memories.len(),
            graph.entities.len(),
            code_symbols.len()
        ),
        GraphNodeKind::Cluster,
    );

    if !memories.is_empty() {
        let kinds = memories
            .iter()
            .map(|memory| memory.kind.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        push_node(
            "overview:memory",
            "Память",
            format!("{} записей · {} типов", memories.len(), kinds),
            GraphNodeKind::Memory,
        );
    }
    if !graph.entities.is_empty() {
        let types = graph
            .entities
            .iter()
            .map(|entity| entity.entity_type.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        push_node(
            "overview:entities",
            "Сущности",
            format!("{} объектов · {} типов", graph.entities.len(), types),
            GraphNodeKind::Entity,
        );
    }
    if !graph.facts.is_empty() {
        push_node(
            "overview:facts",
            "Факты",
            format!("{} фактов", graph.facts.len()),
            GraphNodeKind::Fact,
        );
    }
    if !graph.edges.is_empty() {
        push_node(
            "overview:relations",
            "Связи памяти",
            format!("{} связей", graph.edges.len()),
            GraphNodeKind::Type,
        );
    }

    let memory_clusters = graph
        .entities
        .iter()
        .map(|entity| memory_cluster_for_type(&entity.entity_type).0)
        .collect::<BTreeSet<_>>()
        .len();
    if memory_clusters > 1 {
        push_node(
            "overview:memory-clusters",
            "Области памяти",
            format!("{memory_clusters} кластеров"),
            GraphNodeKind::Cluster,
        );
    }

    if !code_symbols.is_empty() {
        let symbol_kinds = code_symbols
            .iter()
            .map(|symbol| symbol.kind.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        push_node(
            "overview:code",
            "Код",
            format!("{} символов · {} видов", code_symbols.len(), symbol_kinds),
            GraphNodeKind::CodeSymbol,
        );

        let files = code_symbols
            .iter()
            .map(|symbol| symbol.file_path.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        push_node(
            "overview:files",
            "Файлы",
            format!("{files} файлов"),
            GraphNodeKind::CodeFile,
        );

        let directories = code_symbols
            .iter()
            .map(|symbol| code_directory(&symbol.file_path))
            .collect::<BTreeSet<_>>()
            .len();
        if directories > 1 {
            push_node(
                "overview:code-clusters",
                "Области кода",
                format!("{directories} папок"),
                GraphNodeKind::Cluster,
            );
        }
    }

    let mut edges = Vec::<GraphEdge>::new();
    let mut push_edge = |id: &str, from: &str, to: &str, label: &str, weight: usize| {
        if present.contains(from) && present.contains(to) {
            edges.push(GraphEdge {
                id: id.to_string(),
                from: from.to_string(),
                to: to.to_string(),
                label: label.to_string(),
                kind: GraphEdgeKind::Aggregate,
                weight: weight.max(1),
            });
        }
    };

    push_edge(
        "overview-edge:project-memory",
        "overview:project",
        "overview:memory",
        "содержит",
        memories.len(),
    );
    push_edge(
        "overview-edge:memory-entities",
        "overview:memory",
        "overview:entities",
        "описывает",
        graph.entities.len(),
    );
    push_edge(
        "overview-edge:entities-facts",
        "overview:entities",
        "overview:facts",
        "имеет факты",
        graph.facts.len(),
    );
    push_edge(
        "overview-edge:entities-relations",
        "overview:entities",
        "overview:relations",
        "связаны",
        graph.edges.len(),
    );
    push_edge(
        "overview-edge:entities-clusters",
        "overview:entities",
        "overview:memory-clusters",
        "группируются",
        memory_clusters,
    );
    push_edge(
        "overview-edge:project-code",
        "overview:project",
        "overview:code",
        "индексирует",
        code_symbols.len(),
    );
    push_edge(
        "overview-edge:code-files",
        "overview:code",
        "overview:files",
        "лежит в",
        code_symbols.len(),
    );
    push_edge(
        "overview-edge:files-clusters",
        "overview:files",
        "overview:code-clusters",
        "группируются",
        code_symbols.len(),
    );
    push_edge(
        "overview-edge:memory-code",
        "overview:memory",
        "overview:code",
        "связь с кодом",
        memories.len().min(code_symbols.len()),
    );
    if !code_relations.is_empty() {
        push_edge(
            "overview-edge:code-relations",
            "overview:code",
            "overview:files",
            "связи кода",
            code_relations.len(),
        );
    }

    GraphModel {
        nodes,
        edges,
        fallback: false,
    }
}

fn build_relationship_model(memories: &[Memory], graph: &MemoryGraph) -> GraphModel {
    let mut nodes = BTreeMap::new();
    let mut edges = BTreeMap::new();
    let memory_by_id = memories
        .iter()
        .map(|memory| (memory.id.as_str(), memory))
        .collect::<HashMap<_, _>>();

    for entity in &graph.entities {
        nodes.insert(
            entity_node_id(&entity.id),
            GraphNode {
                id: entity_node_id(&entity.id),
                label: entity.name.clone(),
                detail: entity.entity_type.clone(),
                kind: GraphNodeKind::Entity,
            },
        );
    }

    for fact in &graph.facts {
        let fact_id = fact_node_id(&fact.id);
        nodes.insert(
            fact_id.clone(),
            GraphNode {
                id: fact_id.clone(),
                label: truncate_text(
                    &format!("{}: {}", display_relation(&fact.predicate), fact.value),
                    42,
                ),
                detail: fact.value.clone(),
                kind: GraphNodeKind::Fact,
            },
        );
        if let Some(entity_id) = &fact.entity_id {
            let entity_node = entity_node_id(entity_id);
            nodes
                .entry(entity_node.clone())
                .or_insert_with(|| GraphNode {
                    id: entity_node.clone(),
                    label: "Сущность".to_string(),
                    detail: entity_id.clone(),
                    kind: GraphNodeKind::Entity,
                });
            add_edge(
                &mut edges,
                format!("fact-entity:{}", fact.id),
                entity_node,
                fact_id.clone(),
                display_relation(&fact.predicate),
                GraphEdgeKind::Fact,
            );
        }
        if let Some(memory_id) = &fact.memory_id {
            let memory_node = memory_node_id(memory_id);
            ensure_memory_node(
                &mut nodes,
                memory_id,
                memory_by_id.get(memory_id.as_str()).copied(),
            );
            add_edge(
                &mut edges,
                format!("fact-memory:{}", fact.id),
                fact_id,
                memory_node,
                "подтверждено".to_string(),
                GraphEdgeKind::Evidence,
            );
        }
    }

    for edge in &graph.edges {
        let from = entity_node_id(&edge.from_entity_id);
        let to = entity_node_id(&edge.to_entity_id);
        nodes.entry(from.clone()).or_insert_with(|| GraphNode {
            id: from.clone(),
            label: edge.from_entity_name.clone(),
            detail: edge.from_entity_id.clone(),
            kind: GraphNodeKind::Entity,
        });
        nodes.entry(to.clone()).or_insert_with(|| GraphNode {
            id: to.clone(),
            label: edge.to_entity_name.clone(),
            detail: edge.to_entity_id.clone(),
            kind: GraphNodeKind::Entity,
        });
        add_edge(
            &mut edges,
            edge_edge_id(&edge.id),
            from.clone(),
            to.clone(),
            display_relation(&edge.relation_type),
            GraphEdgeKind::Explicit,
        );
        if let Some(memory_id) = &edge.memory_id {
            let memory_node = memory_node_id(memory_id);
            ensure_memory_node(
                &mut nodes,
                memory_id,
                memory_by_id.get(memory_id.as_str()).copied(),
            );
            add_edge(
                &mut edges,
                format!("edge-memory-from:{}", edge.id),
                memory_node.clone(),
                from,
                "основание".to_string(),
                GraphEdgeKind::Evidence,
            );
            add_edge(
                &mut edges,
                format!("edge-memory-to:{}", edge.id),
                memory_node,
                to,
                "основание".to_string(),
                GraphEdgeKind::Evidence,
            );
        }
    }

    let explicit_nodes = nodes.len();
    if explicit_nodes == 0 {
        build_fallback_model(memories)
    } else {
        GraphModel {
            nodes: nodes.into_values().collect(),
            edges: edges.into_values().collect(),
            fallback: false,
        }
    }
}

fn build_type_model(graph: &MemoryGraph) -> GraphModel {
    if graph.entities.is_empty() {
        return GraphModel {
            nodes: Vec::new(),
            edges: Vec::new(),
            fallback: false,
        };
    }

    let mut type_counts = BTreeMap::<String, usize>::new();
    let mut entity_types = HashMap::<String, String>::new();
    for entity in &graph.entities {
        *type_counts.entry(entity.entity_type.clone()).or_default() += 1;
        entity_types.insert(entity.id.clone(), entity.entity_type.clone());
    }

    let nodes = type_counts
        .iter()
        .map(|(entity_type, count)| GraphNode {
            id: type_node_id(entity_type),
            label: entity_type.clone(),
            detail: format!("{count} сущн."),
            kind: GraphNodeKind::Type,
        })
        .collect::<Vec<_>>();

    let mut aggregate = BTreeMap::<(String, String, String), usize>::new();
    for edge in &graph.edges {
        let from_type = entity_types
            .get(&edge.from_entity_id)
            .cloned()
            .unwrap_or_else(|| "concept".to_string());
        let to_type = entity_types
            .get(&edge.to_entity_id)
            .cloned()
            .unwrap_or_else(|| "concept".to_string());
        if from_type == to_type {
            continue;
        }
        *aggregate
            .entry((from_type, edge.relation_type.clone(), to_type))
            .or_default() += 1;
    }

    let edges = aggregate
        .into_iter()
        .map(|((from_type, relation, to_type), weight)| {
            let relation_label = display_relation(&relation);
            GraphEdge {
                id: format!("type-edge:{from_type}:{relation}:{to_type}"),
                from: type_node_id(&from_type),
                to: type_node_id(&to_type),
                label: if weight > 1 {
                    format!("{relation_label} ×{weight}")
                } else {
                    relation_label
                },
                kind: GraphEdgeKind::Aggregate,
                weight,
            }
        })
        .collect::<Vec<_>>();

    GraphModel {
        nodes,
        edges,
        fallback: false,
    }
}

fn build_memory_cluster_model(graph: &MemoryGraph) -> GraphModel {
    if graph.entities.is_empty() {
        return GraphModel {
            nodes: Vec::new(),
            edges: Vec::new(),
            fallback: false,
        };
    }

    let mut cluster_entity_counts = BTreeMap::<String, usize>::new();
    let mut cluster_type_counts = BTreeMap::<String, BTreeSet<String>>::new();
    let mut entity_clusters = HashMap::<String, String>::new();
    for entity in &graph.entities {
        let (cluster_id, _label) = memory_cluster_for_type(&entity.entity_type);
        let cluster_id = cluster_id.to_string();
        *cluster_entity_counts.entry(cluster_id.clone()).or_default() += 1;
        cluster_type_counts
            .entry(cluster_id.clone())
            .or_default()
            .insert(entity.entity_type.clone());
        entity_clusters.insert(entity.id.clone(), cluster_id);
    }

    let nodes = cluster_entity_counts
        .iter()
        .map(|(cluster_id, entity_count)| {
            let type_count = cluster_type_counts
                .get(cluster_id)
                .map(BTreeSet::len)
                .unwrap_or_default();
            GraphNode {
                id: memory_cluster_node_id(cluster_id),
                label: memory_cluster_label(cluster_id).to_string(),
                detail: format!("{entity_count} сущн. · {type_count} типов"),
                kind: GraphNodeKind::Cluster,
            }
        })
        .collect::<Vec<_>>();

    let mut aggregate = BTreeMap::<(String, String, String), usize>::new();
    for edge in &graph.edges {
        let from = entity_clusters
            .get(&edge.from_entity_id)
            .cloned()
            .unwrap_or_else(|| "other".to_string());
        let to = entity_clusters
            .get(&edge.to_entity_id)
            .cloned()
            .unwrap_or_else(|| "other".to_string());
        if from == to {
            continue;
        }
        *aggregate
            .entry((from, edge.relation_type.clone(), to))
            .or_default() += 1;
    }

    GraphModel {
        nodes,
        edges: aggregate
            .into_iter()
            .map(|((from, relation, to), weight)| {
                aggregate_edge(
                    format!("memory-cluster-edge:{from}:{relation}:{to}"),
                    memory_cluster_node_id(&from),
                    memory_cluster_node_id(&to),
                    &relation,
                    GraphEdgeKind::Aggregate,
                    weight,
                )
            })
            .collect(),
        fallback: false,
    }
}

fn build_code_symbol_model(symbols: &[CodeSymbol], relations: &[CodeRelation]) -> GraphModel {
    let mut nodes = BTreeMap::new();
    for symbol in symbols {
        nodes.insert(
            code_node_id(&symbol.id),
            GraphNode {
                id: code_node_id(&symbol.id),
                label: symbol.name.clone(),
                detail: format!(
                    "{} · {}:{}",
                    symbol.kind, symbol.file_path, symbol.start_line
                ),
                kind: GraphNodeKind::CodeSymbol,
            },
        );
    }

    let node_ids = nodes.keys().cloned().collect::<HashSet<_>>();
    let mut aggregate = BTreeMap::<(String, String, String), usize>::new();
    for relation in relations {
        let Some(from_symbol_id) = relation.from_symbol_id.as_deref() else {
            continue;
        };
        let Some(target_symbol_id) = relation.target_symbol_id.as_deref() else {
            continue;
        };
        let from = code_node_id(from_symbol_id);
        let to = code_node_id(target_symbol_id);
        if !node_ids.contains(&from) || !node_ids.contains(&to) {
            continue;
        }
        *aggregate
            .entry((from, relation.relation_kind.clone(), to))
            .or_default() += 1;
    }

    let edges = aggregate
        .into_iter()
        .map(|((from, relation, to), weight)| {
            let relation_label = display_relation(&relation);
            GraphEdge {
                id: format!("code-edge:{from}:{relation}:{to}"),
                from,
                to,
                label: if weight > 1 {
                    format!("{relation_label} ×{weight}")
                } else {
                    relation_label
                },
                kind: GraphEdgeKind::CodeRelation,
                weight,
            }
        })
        .collect();

    GraphModel {
        nodes: nodes.into_values().collect(),
        edges,
        fallback: false,
    }
}

fn build_code_kind_model(symbols: &[CodeSymbol], relations: &[CodeRelation]) -> GraphModel {
    let mut kind_counts = BTreeMap::<String, usize>::new();
    let mut symbol_kinds = HashMap::<String, String>::new();
    for symbol in symbols {
        *kind_counts.entry(symbol.kind.clone()).or_default() += 1;
        symbol_kinds.insert(symbol.id.clone(), symbol.kind.clone());
    }

    let nodes = kind_counts
        .iter()
        .map(|(kind, count)| GraphNode {
            id: code_kind_node_id(kind),
            label: kind.clone(),
            detail: format!("{count} симв."),
            kind: GraphNodeKind::Type,
        })
        .collect::<Vec<_>>();

    let mut aggregate = BTreeMap::<(String, String, String), usize>::new();
    for relation in relations {
        let Some(from_symbol_id) = relation.from_symbol_id.as_deref() else {
            continue;
        };
        let Some(target_symbol_id) = relation.target_symbol_id.as_deref() else {
            continue;
        };
        let Some(from_kind) = symbol_kinds.get(from_symbol_id) else {
            continue;
        };
        let Some(to_kind) = symbol_kinds.get(target_symbol_id) else {
            continue;
        };
        if from_kind == to_kind {
            continue;
        }
        *aggregate
            .entry((
                from_kind.clone(),
                relation.relation_kind.clone(),
                to_kind.clone(),
            ))
            .or_default() += 1;
    }

    let edges = aggregate
        .into_iter()
        .map(|((from_kind, relation, to_kind), weight)| {
            let relation_label = display_relation(&relation);
            GraphEdge {
                id: format!("code-kind-edge:{from_kind}:{relation}:{to_kind}"),
                from: code_kind_node_id(&from_kind),
                to: code_kind_node_id(&to_kind),
                label: if weight > 1 {
                    format!("{relation_label} ×{weight}")
                } else {
                    relation_label
                },
                kind: GraphEdgeKind::Aggregate,
                weight,
            }
        })
        .collect::<Vec<_>>();

    GraphModel {
        nodes,
        edges,
        fallback: false,
    }
}

fn build_code_file_model(symbols: &[CodeSymbol], relations: &[CodeRelation]) -> GraphModel {
    let mut file_counts = BTreeMap::<String, usize>::new();
    let mut symbol_files = HashMap::<String, String>::new();
    for symbol in symbols {
        *file_counts.entry(symbol.file_path.clone()).or_default() += 1;
        symbol_files.insert(symbol.id.clone(), symbol.file_path.clone());
    }

    let nodes = file_counts
        .iter()
        .map(|(file_path, count)| GraphNode {
            id: code_file_node_id(file_path),
            label: file_label(file_path),
            detail: format!("{count} симв. · {file_path}"),
            kind: GraphNodeKind::CodeFile,
        })
        .collect::<Vec<_>>();

    let mut aggregate = BTreeMap::<(String, String, String), usize>::new();
    for relation in relations {
        let Some(from_symbol_id) = relation.from_symbol_id.as_deref() else {
            continue;
        };
        let Some(target_symbol_id) = relation.target_symbol_id.as_deref() else {
            continue;
        };
        let Some(from_file) = symbol_files.get(from_symbol_id) else {
            continue;
        };
        let Some(to_file) = symbol_files.get(target_symbol_id) else {
            continue;
        };
        if from_file == to_file {
            continue;
        }
        *aggregate
            .entry((
                from_file.clone(),
                relation.relation_kind.clone(),
                to_file.clone(),
            ))
            .or_default() += 1;
    }

    let edges = aggregate
        .into_iter()
        .map(|((from_file, relation, to_file), weight)| {
            let relation_label = display_relation(&relation);
            GraphEdge {
                id: format!("code-file-edge:{from_file}:{relation}:{to_file}"),
                from: code_file_node_id(&from_file),
                to: code_file_node_id(&to_file),
                label: if weight > 1 {
                    format!("{relation_label} ×{weight}")
                } else {
                    relation_label
                },
                kind: GraphEdgeKind::Aggregate,
                weight,
            }
        })
        .collect::<Vec<_>>();

    GraphModel {
        nodes,
        edges,
        fallback: false,
    }
}

fn build_code_directory_model(symbols: &[CodeSymbol], relations: &[CodeRelation]) -> GraphModel {
    let mut dir_symbol_counts = BTreeMap::<String, usize>::new();
    let mut dir_files = BTreeMap::<String, BTreeSet<String>>::new();
    let mut symbol_dirs = HashMap::<String, String>::new();
    for symbol in symbols {
        let dir = code_directory(&symbol.file_path);
        *dir_symbol_counts.entry(dir.clone()).or_default() += 1;
        dir_files
            .entry(dir.clone())
            .or_default()
            .insert(symbol.file_path.clone());
        symbol_dirs.insert(symbol.id.clone(), dir);
    }

    let nodes = dir_symbol_counts
        .iter()
        .map(|(dir, symbol_count)| {
            let file_count = dir_files.get(dir).map(BTreeSet::len).unwrap_or_default();
            GraphNode {
                id: code_dir_node_id(dir),
                label: dir.clone(),
                detail: format!("{file_count} файлов · {symbol_count} симв."),
                kind: GraphNodeKind::Cluster,
            }
        })
        .collect::<Vec<_>>();

    let mut aggregate = BTreeMap::<(String, String, String), usize>::new();
    for relation in relations {
        let Some(from_symbol_id) = relation.from_symbol_id.as_deref() else {
            continue;
        };
        let Some(target_symbol_id) = relation.target_symbol_id.as_deref() else {
            continue;
        };
        let Some(from_dir) = symbol_dirs.get(from_symbol_id) else {
            continue;
        };
        let Some(to_dir) = symbol_dirs.get(target_symbol_id) else {
            continue;
        };
        if from_dir == to_dir {
            continue;
        }
        *aggregate
            .entry((
                from_dir.clone(),
                relation.relation_kind.clone(),
                to_dir.clone(),
            ))
            .or_default() += 1;
    }

    GraphModel {
        nodes,
        edges: aggregate
            .into_iter()
            .map(|((from, relation, to), weight)| {
                aggregate_edge(
                    format!("code-dir-edge:{from}:{relation}:{to}"),
                    code_dir_node_id(&from),
                    code_dir_node_id(&to),
                    &relation,
                    GraphEdgeKind::Aggregate,
                    weight,
                )
            })
            .collect(),
        fallback: false,
    }
}

fn build_bridge_model(
    memories: &[Memory],
    graph: &MemoryGraph,
    symbols: &[CodeSymbol],
) -> GraphModel {
    let mut nodes = BTreeMap::<String, GraphNode>::new();
    let mut edges = BTreeMap::<String, GraphEdge>::new();
    let target_index = CodeTargetIndex::new(symbols);
    let memory_by_id = memories
        .iter()
        .map(|memory| (memory.id.as_str(), memory))
        .collect::<HashMap<_, _>>();

    for memory in memories {
        let memory_node = memory_node_id(&memory.id);
        ensure_memory_node(&mut nodes, &memory.id, Some(memory));
        for target in target_index.targets_for_text(&memory_search_text(memory), 8) {
            ensure_code_target_node(&mut nodes, &target);
            add_edge(
                &mut edges,
                format!("bridge-memory-code:{}:{}", memory.id, target.id),
                memory_node.clone(),
                target.id,
                "упоминает код".to_string(),
                GraphEdgeKind::Derived,
            );
        }
    }

    for fact in &graph.facts {
        let Some(memory_id) = fact.memory_id.as_deref() else {
            continue;
        };
        let Some(entity_id) = fact.entity_id.as_deref() else {
            continue;
        };
        let memory_node = memory_node_id(memory_id);
        let entity_node = entity_node_id(entity_id);
        ensure_memory_node(&mut nodes, memory_id, memory_by_id.get(memory_id).copied());
        if let Some(entity) = graph.entities.iter().find(|entity| entity.id == entity_id) {
            ensure_entity_graph_node(&mut nodes, entity);
        }
        add_edge(
            &mut edges,
            format!("bridge-fact:{}:{entity_id}", fact.id),
            memory_node,
            entity_node,
            display_relation(&fact.predicate),
            GraphEdgeKind::Fact,
        );
    }

    for entity in &graph.entities {
        let entity_node = entity_node_id(&entity.id);
        let text = entity_search_text(entity);
        for target in target_index.targets_for_text(&text, 5) {
            ensure_entity_graph_node(&mut nodes, entity);
            ensure_code_target_node(&mut nodes, &target);
            add_edge(
                &mut edges,
                format!("bridge-entity-code:{}:{}", entity.id, target.id),
                entity_node.clone(),
                target.id,
                "связано с кодом".to_string(),
                GraphEdgeKind::Derived,
            );
        }
    }

    GraphModel {
        nodes: nodes.into_values().collect(),
        edges: edges.into_values().collect(),
        fallback: false,
    }
}

fn build_bridge_cluster_model(
    memories: &[Memory],
    graph: &MemoryGraph,
    symbols: &[CodeSymbol],
) -> GraphModel {
    let mut nodes = BTreeMap::<String, GraphNode>::new();
    let mut aggregate = BTreeMap::<(String, String, String), usize>::new();
    let target_index = CodeTargetIndex::new(symbols);

    for memory in memories {
        let from = memory_kind_cluster_node_id(&memory.kind);
        nodes.entry(from.clone()).or_insert_with(|| GraphNode {
            id: from.clone(),
            label: format!("Память: {}", memory.kind),
            detail: "тип памяти".to_string(),
            kind: GraphNodeKind::Cluster,
        });
        for target in target_index.targets_for_text(&memory_search_text(memory), 8) {
            let dir = target
                .file_path
                .as_deref()
                .map(code_directory)
                .unwrap_or_else(|| "код".to_string());
            let to = code_dir_node_id(&dir);
            nodes.entry(to.clone()).or_insert_with(|| GraphNode {
                id: to.clone(),
                label: dir.clone(),
                detail: "кластер кода".to_string(),
                kind: GraphNodeKind::Cluster,
            });
            *aggregate
                .entry((from.clone(), "mentions_code".to_string(), to))
                .or_default() += 1;
        }
    }

    for entity in &graph.entities {
        let (cluster, label) = memory_cluster_for_type(&entity.entity_type);
        let from = memory_cluster_node_id(cluster);
        nodes.entry(from.clone()).or_insert_with(|| GraphNode {
            id: from.clone(),
            label: label.to_string(),
            detail: "кластер сущностей памяти".to_string(),
            kind: GraphNodeKind::Cluster,
        });
        for target in target_index.targets_for_text(&entity_search_text(entity), 5) {
            let dir = target
                .file_path
                .as_deref()
                .map(code_directory)
                .unwrap_or_else(|| "код".to_string());
            let to = code_dir_node_id(&dir);
            nodes.entry(to.clone()).or_insert_with(|| GraphNode {
                id: to.clone(),
                label: dir.clone(),
                detail: "кластер кода".to_string(),
                kind: GraphNodeKind::Cluster,
            });
            *aggregate
                .entry((from.clone(), "relates_code".to_string(), to))
                .or_default() += 1;
        }
    }

    GraphModel {
        nodes: nodes.into_values().collect(),
        edges: aggregate
            .into_iter()
            .map(|((from, relation, to), weight)| {
                aggregate_edge(
                    format!("bridge-cluster-edge:{from}:{relation}:{to}"),
                    from,
                    to,
                    &relation,
                    GraphEdgeKind::Aggregate,
                    weight,
                )
            })
            .collect(),
        fallback: false,
    }
}

fn build_memory_cluster_drilldown_model(
    memories: &[Memory],
    graph: &MemoryGraph,
    cluster_id: &str,
) -> Option<GraphModel> {
    let cluster = cluster_id.strip_prefix("memory-cluster:")?;
    let seeds = graph
        .entities
        .iter()
        .filter(|entity| memory_cluster_for_type(&entity.entity_type).0 == cluster)
        .map(|entity| entity_node_id(&entity.id))
        .collect::<HashSet<_>>();
    drilldown_neighborhood(build_relationship_model(memories, graph), seeds)
}

fn build_code_cluster_drilldown_model(
    symbols: &[CodeSymbol],
    relations: &[CodeRelation],
    cluster_id: &str,
) -> Option<GraphModel> {
    let dir = cluster_id.strip_prefix("code-dir:")?;
    let seeds = symbols
        .iter()
        .filter(|symbol| code_directory(&symbol.file_path) == dir)
        .map(|symbol| code_file_node_id(&symbol.file_path))
        .collect::<HashSet<_>>();
    drilldown_neighborhood(build_code_file_model(symbols, relations), seeds)
}

fn build_bridge_cluster_drilldown_model(
    memories: &[Memory],
    graph: &MemoryGraph,
    symbols: &[CodeSymbol],
    cluster_id: &str,
) -> Option<GraphModel> {
    let mut seeds = HashSet::<String>::new();
    if let Some(kind) = cluster_id.strip_prefix("memory-kind:") {
        seeds.extend(
            memories
                .iter()
                .filter(|memory| memory.kind == kind)
                .map(|memory| memory_node_id(&memory.id)),
        );
    } else if let Some(cluster) = cluster_id.strip_prefix("memory-cluster:") {
        seeds.extend(
            graph
                .entities
                .iter()
                .filter(|entity| memory_cluster_for_type(&entity.entity_type).0 == cluster)
                .map(|entity| entity_node_id(&entity.id)),
        );
    } else if let Some(dir) = cluster_id.strip_prefix("code-dir:") {
        for symbol in symbols
            .iter()
            .filter(|symbol| code_directory(&symbol.file_path) == dir)
        {
            seeds.insert(code_file_node_id(&symbol.file_path));
            seeds.insert(code_node_id(&symbol.id));
        }
    }
    drilldown_neighborhood(build_bridge_model(memories, graph, symbols), seeds)
}

fn drilldown_neighborhood(model: GraphModel, seeds: HashSet<String>) -> Option<GraphModel> {
    if seeds.is_empty() {
        return None;
    }
    let mut allowed = seeds.clone();
    for edge in &model.edges {
        if seeds.contains(&edge.from) || seeds.contains(&edge.to) {
            allowed.insert(edge.from.clone());
            allowed.insert(edge.to.clone());
        }
    }

    let nodes = model
        .nodes
        .into_iter()
        .filter(|node| allowed.contains(&node.id))
        .collect::<Vec<_>>();
    if nodes.is_empty() {
        return None;
    }
    let node_ids = nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<HashSet<_>>();
    let edges = model
        .edges
        .into_iter()
        .filter(|edge| node_ids.contains(&edge.from) && node_ids.contains(&edge.to))
        .collect::<Vec<_>>();
    Some(GraphModel {
        nodes,
        edges,
        fallback: model.fallback,
    })
}

fn build_fallback_model(memories: &[Memory]) -> GraphModel {
    let mut nodes = BTreeMap::new();
    let mut edges = BTreeMap::new();
    for memory in memories {
        ensure_memory_node(&mut nodes, &memory.id, Some(memory));
    }

    let memory_by_id = memories
        .iter()
        .map(|memory| (memory.id.as_str(), memory))
        .collect::<HashMap<_, _>>();
    for memory in memories {
        if let Some(superseded_by) = &memory.superseded_by
            && memory_by_id.contains_key(superseded_by.as_str())
        {
            add_edge(
                &mut edges,
                format!("derived-superseded:{}:{superseded_by}", memory.id),
                memory_node_id(&memory.id),
                memory_node_id(superseded_by),
                "заменена".to_string(),
                GraphEdgeKind::Derived,
            );
        }
    }

    let mut tag_index: BTreeMap<String, Vec<&Memory>> = BTreeMap::new();
    for memory in memories {
        for tag in &memory.tags {
            tag_index.entry(tag.clone()).or_default().push(memory);
        }
    }
    for (tag, tagged) in tag_index {
        if tagged.len() < 2 || tagged.len() > 8 {
            continue;
        }
        for pair in tagged.windows(2) {
            add_edge(
                &mut edges,
                format!("derived-tag:{}:{}:{}", tag, pair[0].id, pair[1].id),
                memory_node_id(&pair[0].id),
                memory_node_id(&pair[1].id),
                format!("#{tag}"),
                GraphEdgeKind::Derived,
            );
        }
    }

    let mut source_index: BTreeMap<String, Vec<&Memory>> = BTreeMap::new();
    for memory in memories {
        if let Some(source) = &memory.source {
            source_index.entry(source.clone()).or_default().push(memory);
        }
    }
    for (source, sourced) in source_index {
        if sourced.len() < 2 || sourced.len() > 6 {
            continue;
        }
        for pair in sourced.windows(2) {
            add_edge(
                &mut edges,
                format!("derived-source:{}:{}:{}", source, pair[0].id, pair[1].id),
                memory_node_id(&pair[0].id),
                memory_node_id(&pair[1].id),
                "источник".to_string(),
                GraphEdgeKind::Derived,
            );
        }
    }

    GraphModel {
        nodes: nodes.into_values().collect(),
        edges: edges.into_values().collect(),
        fallback: true,
    }
}

struct RelationshipFilterOptions<'a> {
    selection: &'a GraphSelection,
    show_facts: bool,
    show_evidence: bool,
    focus_selected: bool,
    max_visible_nodes: usize,
    node_filter: &'a str,
    relation_filter: &'a str,
}

fn filter_relationship_model(
    model: GraphModel,
    options: RelationshipFilterOptions<'_>,
) -> GraphModel {
    let fallback = model.fallback;
    let mut nodes = model
        .nodes
        .into_iter()
        .filter(|node| options.show_facts || node.kind != GraphNodeKind::Fact)
        .filter(|node| fallback || options.show_evidence || node.kind != GraphNodeKind::Memory)
        .collect::<Vec<_>>();
    let node_ids = nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<HashSet<_>>();
    let mut edges = model
        .edges
        .into_iter()
        .filter(|edge| options.show_facts || edge.kind != GraphEdgeKind::Fact)
        .filter(|edge| options.show_evidence || edge.kind != GraphEdgeKind::Evidence)
        .filter(|edge| node_ids.contains(&edge.from) && node_ids.contains(&edge.to))
        .collect::<Vec<_>>();

    let mut filtered = GraphModel {
        nodes: std::mem::take(&mut nodes),
        edges: std::mem::take(&mut edges),
        fallback,
    };

    apply_text_filters(&mut filtered, options.node_filter, options.relation_filter);

    if options.focus_selected && !matches!(options.selection, GraphSelection::Overview) {
        let focus_ids = selected_neighbor_nodes(&filtered, options.selection);
        if !focus_ids.is_empty() {
            filtered.nodes.retain(|node| focus_ids.contains(&node.id));
            filtered
                .edges
                .retain(|edge| focus_ids.contains(&edge.from) && focus_ids.contains(&edge.to));
        }
    }

    let filtered = clamp_visible_nodes(filtered, options.selection, options.max_visible_nodes);
    clamp_visible_edges(filtered, options.selection, options.max_visible_nodes)
}

fn apply_text_filters(model: &mut GraphModel, node_filter: &str, relation_filter: &str) {
    let node_filter = node_filter.trim().to_ascii_lowercase();
    let relation_filter = relation_filter.trim().to_ascii_lowercase();

    if !relation_filter.is_empty() {
        model
            .edges
            .retain(|edge| edge.label.to_ascii_lowercase().contains(&relation_filter));
        let connected = model
            .edges
            .iter()
            .flat_map(|edge| [edge.from.clone(), edge.to.clone()])
            .collect::<HashSet<_>>();
        model.nodes.retain(|node| connected.contains(&node.id));
    }

    if !node_filter.is_empty() {
        let matches = model
            .nodes
            .iter()
            .filter(|node| {
                node.label.to_ascii_lowercase().contains(&node_filter)
                    || node.detail.to_ascii_lowercase().contains(&node_filter)
            })
            .map(|node| node.id.clone())
            .collect::<HashSet<_>>();
        if matches.is_empty() {
            model.nodes.clear();
            model.edges.clear();
            return;
        }

        let mut visible = matches.clone();
        for edge in &model.edges {
            if matches.contains(&edge.from) {
                visible.insert(edge.to.clone());
            }
            if matches.contains(&edge.to) {
                visible.insert(edge.from.clone());
            }
        }
        model.nodes.retain(|node| visible.contains(&node.id));
        model
            .edges
            .retain(|edge| visible.contains(&edge.from) && visible.contains(&edge.to));
    }
}

fn clamp_visible_nodes(
    mut model: GraphModel,
    selection: &GraphSelection,
    max_visible_nodes: usize,
) -> GraphModel {
    let max_visible_nodes = max_visible_nodes.clamp(20, 250);
    if model.nodes.len() <= max_visible_nodes {
        return model;
    }

    let selected_ids = selected_neighbor_nodes(&model, selection);
    let mut degree = HashMap::<String, usize>::new();
    for edge in &model.edges {
        *degree.entry(edge.from.clone()).or_default() += 1;
        *degree.entry(edge.to.clone()).or_default() += 1;
    }

    model.nodes.sort_by(|left, right| {
        node_priority(right, &degree, &selected_ids, selection)
            .cmp(&node_priority(left, &degree, &selected_ids, selection))
            .then_with(|| left.label.cmp(&right.label))
    });
    model.nodes.truncate(max_visible_nodes);
    let kept = model
        .nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<HashSet<_>>();
    model
        .edges
        .retain(|edge| kept.contains(&edge.from) && kept.contains(&edge.to));
    model
}

fn clamp_visible_edges(
    mut model: GraphModel,
    selection: &GraphSelection,
    max_visible_nodes: usize,
) -> GraphModel {
    let max_visible_edges = (max_visible_nodes * 5).clamp(80, 700);
    if model.edges.len() <= max_visible_edges {
        return model;
    }

    let selected_ids = selected_neighbor_nodes(&model, selection);
    model.edges.sort_by(|left, right| {
        edge_priority(right, &selected_ids, selection)
            .cmp(&edge_priority(left, &selected_ids, selection))
            .then_with(|| left.label.cmp(&right.label))
            .then_with(|| left.id.cmp(&right.id))
    });
    model.edges.truncate(max_visible_edges);
    model
}

fn edge_priority(
    edge: &GraphEdge,
    selected_ids: &HashSet<String>,
    selection: &GraphSelection,
) -> i32 {
    let mut score = edge.weight.max(1) as i32 * 20;
    if selected_ids.contains(&edge.from) || selected_ids.contains(&edge.to) {
        score += 2_000;
    }
    if matches!(selection, GraphSelection::Edge(id) if id == &edge.id) {
        score += 12_000;
    }
    score
}

fn node_priority(
    node: &GraphNode,
    degree: &HashMap<String, usize>,
    selected_ids: &HashSet<String>,
    selection: &GraphSelection,
) -> i32 {
    let mut score = degree.get(&node.id).copied().unwrap_or_default() as i32 * 12;
    score += match node.kind {
        GraphNodeKind::Cluster => 620,
        GraphNodeKind::Entity => 400,
        GraphNodeKind::Type => 520,
        GraphNodeKind::CodeSymbol => 430,
        GraphNodeKind::CodeFile => 560,
        GraphNodeKind::Fact => 180,
        GraphNodeKind::Memory => 80,
    };
    if selected_ids.contains(&node.id) {
        score += 8_000;
    }
    if matches!(selection, GraphSelection::Node(id) if id == &node.id) {
        score += 12_000;
    }
    score
}

fn ensure_memory_node(
    nodes: &mut BTreeMap<String, GraphNode>,
    memory_id: &str,
    memory: Option<&Memory>,
) {
    let id = memory_node_id(memory_id);
    nodes.entry(id.clone()).or_insert_with(|| GraphNode {
        id,
        label: memory
            .map(|memory| truncate_text(memory.body.trim(), 48))
            .unwrap_or_else(|| "Память".to_string()),
        detail: memory
            .map(|memory| memory.kind.clone())
            .unwrap_or_else(|| memory_id.to_string()),
        kind: GraphNodeKind::Memory,
    });
}

fn add_edge(
    edges: &mut BTreeMap<String, GraphEdge>,
    id: String,
    from: String,
    to: String,
    label: String,
    kind: GraphEdgeKind,
) {
    if from == to {
        return;
    }
    edges.entry(id.clone()).or_insert(GraphEdge {
        id,
        from,
        to,
        label,
        kind,
        weight: 1,
    });
}

fn aggregate_edge(
    id: String,
    from: String,
    to: String,
    relation: &str,
    kind: GraphEdgeKind,
    weight: usize,
) -> GraphEdge {
    let relation_label = display_relation(relation);
    GraphEdge {
        id,
        from,
        to,
        label: if weight > 1 {
            format!("{relation_label} ×{weight}")
        } else {
            relation_label
        },
        kind,
        weight,
    }
}

fn ensure_entity_graph_node(
    nodes: &mut BTreeMap<String, GraphNode>,
    entity: &crate::store::MemoryEntity,
) {
    let id = entity_node_id(&entity.id);
    nodes.entry(id.clone()).or_insert_with(|| GraphNode {
        id,
        label: entity.name.clone(),
        detail: entity.entity_type.clone(),
        kind: GraphNodeKind::Entity,
    });
}

fn ensure_code_target_node(nodes: &mut BTreeMap<String, GraphNode>, target: &CodeTarget) {
    nodes.entry(target.id.clone()).or_insert_with(|| GraphNode {
        id: target.id.clone(),
        label: target.label.clone(),
        detail: target.detail.clone(),
        kind: target.kind,
    });
}

fn layout_graph(
    model: &GraphModel,
    rect: Rect,
    selection: &GraphSelection,
) -> BTreeMap<String, Pos2> {
    let mut positions = BTreeMap::new();
    let count = model.nodes.len();
    if count == 0 {
        return positions;
    }
    let center = rect.center();
    let base_radius = rect.width().min(rect.height()) * 0.33;
    let focus_id = match selection {
        GraphSelection::Node(id) if model.nodes.iter().any(|node| &node.id == id) => {
            Some(id.clone())
        }
        _ => None,
    };

    if focus_id.is_none() && is_project_overview_model(model) {
        layout_project_overview(model, rect, &mut positions);
        resolve_node_overlaps(model, rect, &mut positions, Some("overview:project"));
        return positions;
    } else if focus_id.is_none() {
        layout_radial_card_rings(model, rect, selection, &mut positions);
        resolve_node_overlaps(model, rect, &mut positions, None);
        return positions;
    } else if let Some(focus_id) = &focus_id {
        positions.insert(focus_id.clone(), center);
        let direct_neighbors = model
            .edges
            .iter()
            .filter_map(|edge| {
                if &edge.from == focus_id {
                    Some(edge.to.clone())
                } else if &edge.to == focus_id {
                    Some(edge.from.clone())
                } else {
                    None
                }
            })
            .collect::<BTreeSet<_>>();
        for (ring_index, node_id) in direct_neighbors.iter().enumerate() {
            let angle = TAU * ring_index as f32 / direct_neighbors.len().max(1) as f32;
            positions.insert(
                node_id.clone(),
                Pos2::new(
                    center.x + angle.cos() * base_radius * 0.72,
                    center.y + angle.sin() * base_radius * 0.72,
                ),
            );
        }
        let others = model
            .nodes
            .iter()
            .filter(|node| &node.id != focus_id && !direct_neighbors.contains(&node.id))
            .collect::<Vec<_>>();
        for (index, node) in others.iter().enumerate() {
            let angle = TAU * index as f32 / others.len().max(1) as f32;
            positions.insert(
                node.id.clone(),
                Pos2::new(
                    center.x + angle.cos() * base_radius * 1.18,
                    center.y + angle.sin() * base_radius * 1.18,
                ),
            );
        }
    } else {
        for (index, node) in model.nodes.iter().enumerate() {
            let angle = TAU * index as f32 / count as f32;
            let radius = match node.kind {
                GraphNodeKind::Cluster => base_radius * 0.48,
                GraphNodeKind::Type => base_radius * 0.58,
                GraphNodeKind::CodeFile => base_radius * 0.62,
                GraphNodeKind::Entity => base_radius * 0.68,
                GraphNodeKind::CodeSymbol => base_radius * 0.78,
                GraphNodeKind::Fact => base_radius * 0.92,
                GraphNodeKind::Memory => base_radius * 1.18,
            };
            positions.insert(
                node.id.clone(),
                Pos2::new(
                    center.x + angle.cos() * radius,
                    center.y + angle.sin() * radius,
                ),
            );
        }
    }

    let node_ids = model
        .nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<Vec<_>>();
    let edge_pairs = model
        .edges
        .iter()
        .map(|edge| (edge.from.clone(), edge.to.clone()))
        .collect::<Vec<_>>();

    let iterations = layout_iterations(count);
    let repulsion_stride = layout_repulsion_stride(count);
    for _ in 0..iterations {
        let mut forces = node_ids
            .iter()
            .map(|id| (id.clone(), Vec2::ZERO))
            .collect::<BTreeMap<_, _>>();

        for i in 0..node_ids.len() {
            for j in ((i + 1)..node_ids.len()).step_by(repulsion_stride) {
                let a = &node_ids[i];
                let b = &node_ids[j];
                let delta = positions[a] - positions[b];
                let distance = delta.length().max(12.0);
                let direction = delta / distance;
                let force = (9000.0 / (distance * distance)).min(5.0);
                *forces.get_mut(a).expect("force exists") += direction * force;
                *forces.get_mut(b).expect("force exists") -= direction * force;
            }
        }

        for (from, to) in &edge_pairs {
            let Some(from_pos) = positions.get(from) else {
                continue;
            };
            let Some(to_pos) = positions.get(to) else {
                continue;
            };
            let delta = *to_pos - *from_pos;
            let distance = delta.length().max(1.0);
            let direction = delta / distance;
            let target = 180.0;
            let force = (distance - target) * 0.018;
            *forces.get_mut(from).expect("force exists") += direction * force;
            *forces.get_mut(to).expect("force exists") -= direction * force;
        }

        for id in &node_ids {
            let force = forces[id];
            let pos = positions.get_mut(id).expect("position exists");
            if focus_id.as_deref() == Some(id.as_str()) {
                *pos = center;
                continue;
            }
            pos.x = (pos.x + force.x).clamp(rect.left() + 90.0, rect.right() - 90.0);
            pos.y = (pos.y + force.y).clamp(rect.top() + 80.0, rect.bottom() - 80.0);
        }
    }

    resolve_node_overlaps(model, rect, &mut positions, focus_id.as_deref());
    positions
}

fn layout_fixed_node_id<'a>(
    model: &'a GraphModel,
    selection: &'a GraphSelection,
) -> Option<&'a str> {
    match selection {
        GraphSelection::Node(id) if model.nodes.iter().any(|node| &node.id == id) => {
            Some(id.as_str())
        }
        _ if is_project_overview_model(model) => Some("overview:project"),
        _ => None,
    }
}

fn is_project_overview_model(model: &GraphModel) -> bool {
    model.nodes.iter().any(|node| node.id == "overview:project")
}

fn layout_radial_card_rings(
    model: &GraphModel,
    rect: Rect,
    selection: &GraphSelection,
    positions: &mut BTreeMap<String, Pos2>,
) {
    let mut nodes = model.nodes.iter().collect::<Vec<_>>();
    let selected_neighbors = selected_neighbor_nodes(model, selection);
    nodes.sort_by(|left, right| {
        radial_layout_priority(right, selection, &selected_neighbors)
            .cmp(&radial_layout_priority(
                left,
                selection,
                &selected_neighbors,
            ))
            .then_with(|| left.kind.label().cmp(right.kind.label()))
            .then_with(|| left.label.cmp(&right.label))
            .then_with(|| left.id.cmp(&right.id))
    });

    let max_size = nodes
        .iter()
        .map(|node| layout_node_size(node.kind))
        .fold(Vec2::ZERO, |acc, size| {
            Vec2::new(acc.x.max(size.x), acc.y.max(size.y))
        });

    let center = rect.center();
    let arc_spacing = (max_size.x + NODE_COLLISION_PADDING * 2.2).max(180.0);
    let ring_gap = (max_size.y + NODE_COLLISION_PADDING * 2.6).max(92.0);
    let mut radius = (arc_spacing * 6.0 / TAU).max(max_size.x * 0.95);
    let mut index = 0;
    let mut ring_index = 0;

    while index < nodes.len() {
        let remaining = nodes.len() - index;
        let capacity = ((TAU * radius) / arc_spacing).floor().max(6.0) as usize;
        let take = remaining.min(capacity);
        let phase = ring_index as f32 * 0.47;

        for slot in 0..take {
            let node = nodes[index + slot];
            let angle = phase + TAU * slot as f32 / take.max(1) as f32;
            let kind_offset = match node.kind {
                GraphNodeKind::Cluster => -ring_gap * 0.18,
                GraphNodeKind::CodeFile => -ring_gap * 0.1,
                GraphNodeKind::Memory => -ring_gap * 0.04,
                GraphNodeKind::CodeSymbol => ring_gap * 0.02,
                GraphNodeKind::Entity => ring_gap * 0.06,
                GraphNodeKind::Type => ring_gap * 0.1,
                GraphNodeKind::Fact => ring_gap * 0.14,
            };
            let node_radius = (radius + kind_offset).max(max_size.x * 0.8);
            positions.insert(
                node.id.clone(),
                Pos2::new(
                    center.x + angle.cos() * node_radius,
                    center.y + angle.sin() * node_radius,
                ),
            );
        }

        index += take;
        ring_index += 1;
        radius += ring_gap;
    }
}

fn radial_layout_priority(
    node: &GraphNode,
    selection: &GraphSelection,
    selected_neighbors: &HashSet<String>,
) -> usize {
    let mut score = node_kind_priority(node.kind);
    if selected_neighbors.contains(&node.id) {
        score += 5_000;
    }
    if matches!(selection, GraphSelection::Node(id) if id == &node.id) {
        score += 10_000;
    }
    score
}

fn node_kind_priority(kind: GraphNodeKind) -> usize {
    match kind {
        GraphNodeKind::Cluster => 700,
        GraphNodeKind::CodeFile => 650,
        GraphNodeKind::Memory => 620,
        GraphNodeKind::CodeSymbol => 560,
        GraphNodeKind::Entity => 520,
        GraphNodeKind::Type => 480,
        GraphNodeKind::Fact => 360,
    }
}

fn layout_project_overview(model: &GraphModel, rect: Rect, positions: &mut BTreeMap<String, Pos2>) {
    let center = rect.center();
    let width = rect.width().min(1120.0);
    let height = rect.height().min(720.0);
    let dx = width * 0.25;
    let far_dx = width * 0.38;
    let dy = height * 0.22;
    let far_dy = height * 0.34;
    let anchors = [
        ("overview:project", Vec2::ZERO),
        ("overview:memory", Vec2::new(-dx, 0.0)),
        ("overview:entities", Vec2::new(-dx, -dy)),
        ("overview:facts", Vec2::new(-far_dx, -far_dy)),
        ("overview:relations", Vec2::new(-far_dx, dy)),
        ("overview:memory-clusters", Vec2::new(-dx, far_dy)),
        ("overview:code", Vec2::new(dx, 0.0)),
        ("overview:files", Vec2::new(far_dx, -dy)),
        ("overview:code-clusters", Vec2::new(far_dx, far_dy)),
    ];
    for (node_id, offset) in anchors {
        if model.nodes.iter().any(|node| node.id == node_id) {
            positions.insert(node_id.to_string(), center + offset);
        }
    }

    let mut overflow = model
        .nodes
        .iter()
        .filter(|node| !positions.contains_key(&node.id))
        .collect::<Vec<_>>();
    overflow.sort_by(|a, b| a.id.cmp(&b.id));
    let radius = rect.width().min(rect.height()) * 0.42;
    for (index, node) in overflow.iter().enumerate() {
        let angle = TAU * index as f32 / overflow.len().max(1) as f32;
        positions.insert(
            node.id.clone(),
            Pos2::new(
                center.x + angle.cos() * radius,
                center.y + angle.sin() * radius,
            ),
        );
    }
}

fn resolve_node_overlaps(
    model: &GraphModel,
    rect: Rect,
    positions: &mut BTreeMap<String, Pos2>,
    fixed_id: Option<&str>,
) {
    let nodes = model
        .nodes
        .iter()
        .filter(|node| positions.contains_key(&node.id))
        .collect::<Vec<_>>();
    if nodes.len() < 2 {
        return;
    }

    let iterations = collision_iterations(nodes.len());
    for _ in 0..iterations {
        let mut moved = false;
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let a = nodes[i];
                let b = nodes[j];
                let a_pos = positions[&a.id];
                let b_pos = positions[&b.id];
                let minimum = node_separation(a.kind, b.kind, MIN_GRAPH_ZOOM);
                let delta = b_pos - a_pos;
                let overlap_x = minimum.x - delta.x.abs();
                let overlap_y = minimum.y - delta.y.abs();
                if overlap_x <= 0.0 || overlap_y <= 0.0 {
                    continue;
                }

                let a_fixed = fixed_id == Some(a.id.as_str());
                let b_fixed = fixed_id == Some(b.id.as_str());
                if a_fixed && b_fixed {
                    continue;
                }

                let shift = if overlap_x < overlap_y {
                    let sign = signed_axis(delta.x, i, j);
                    Vec2::new(sign * (overlap_x + 0.8), 0.0)
                } else {
                    let sign = signed_axis(delta.y, i, j);
                    Vec2::new(0.0, sign * (overlap_y + 0.8))
                };

                if a_fixed {
                    *positions.get_mut(&b.id).expect("position exists") += shift;
                } else if b_fixed {
                    *positions.get_mut(&a.id).expect("position exists") -= shift;
                } else {
                    *positions.get_mut(&a.id).expect("position exists") -= shift * 0.5;
                    *positions.get_mut(&b.id).expect("position exists") += shift * 0.5;
                }
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }

    if fixed_id.is_none() {
        recenter_layout(rect, positions);
    }
}

fn resolve_node_offset_overlaps(
    model: &GraphModel,
    positions: &BTreeMap<String, Pos2>,
    offsets: &mut BTreeMap<String, Vec2>,
    zoom: f32,
    fixed_id: Option<&str>,
    pinned_nodes: &BTreeSet<String>,
) {
    let nodes = model
        .nodes
        .iter()
        .filter(|node| positions.contains_key(&node.id))
        .collect::<Vec<_>>();
    if nodes.len() < 2 {
        return;
    }

    let zoom = zoom.clamp(MIN_GRAPH_ZOOM, MAX_GRAPH_ZOOM);
    let iterations = (collision_iterations(nodes.len()) * 2).clamp(80, 180);
    for _ in 0..iterations {
        let mut moved = false;
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let a = nodes[i];
                let b = nodes[j];
                let a_pos = positions[&a.id] + offsets.get(&a.id).copied().unwrap_or_default();
                let b_pos = positions[&b.id] + offsets.get(&b.id).copied().unwrap_or_default();
                let minimum = node_separation(a.kind, b.kind, zoom);
                let delta = b_pos - a_pos;
                let overlap_x = minimum.x - delta.x.abs();
                let overlap_y = minimum.y - delta.y.abs();
                if overlap_x <= 0.0 || overlap_y <= 0.0 {
                    continue;
                }

                let shift = if overlap_x < overlap_y {
                    let sign = signed_axis(delta.x, i, j);
                    Vec2::new(sign * (overlap_x + 0.9), 0.0)
                } else {
                    let sign = signed_axis(delta.y, i, j);
                    Vec2::new(0.0, sign * (overlap_y + 0.9))
                };

                let a_fixed = fixed_id == Some(a.id.as_str());
                let b_fixed = fixed_id == Some(b.id.as_str());
                let a_pinned = pinned_nodes.contains(&a.id);
                let b_pinned = pinned_nodes.contains(&b.id);
                if a_fixed && b_fixed {
                    continue;
                } else if a_fixed || (a_pinned && !b_pinned && !b_fixed) {
                    move_node_offset(offsets, &b.id, shift);
                } else if b_fixed || (b_pinned && !a_pinned && !a_fixed) {
                    move_node_offset(offsets, &a.id, -shift);
                } else {
                    move_node_offset(offsets, &a.id, -shift * 0.5);
                    move_node_offset(offsets, &b.id, shift * 0.5);
                }
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
}

fn move_node_offset(offsets: &mut BTreeMap<String, Vec2>, node_id: &str, delta: Vec2) {
    if delta.x.abs() + delta.y.abs() < 0.01 {
        return;
    }
    *offsets.entry(node_id.to_string()).or_insert(Vec2::ZERO) += delta;
}

fn layout_node_size(kind: GraphNodeKind) -> Vec2 {
    collision_node_size(kind, MIN_GRAPH_ZOOM)
}

fn node_separation(a: GraphNodeKind, b: GraphNodeKind, zoom: f32) -> Vec2 {
    let zoom = zoom.max(MIN_GRAPH_ZOOM);
    (collision_node_size(a, zoom) + collision_node_size(b, zoom)) * 0.5
        + Vec2::splat(NODE_COLLISION_PADDING / zoom)
}

fn collision_node_size(kind: GraphNodeKind, zoom: f32) -> Vec2 {
    node_card_size(kind) / zoom.max(MIN_GRAPH_ZOOM)
}

fn signed_axis(delta: f32, i: usize, j: usize) -> f32 {
    if delta.abs() > 0.1 {
        delta.signum()
    } else if (i + j).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

fn collision_iterations(node_count: usize) -> usize {
    match node_count {
        0..=30 => 64,
        31..=80 => 84,
        81..=160 => 72,
        _ => 56,
    }
}

fn recenter_layout(rect: Rect, positions: &mut BTreeMap<String, Pos2>) {
    let Some(bounds) = graph_world_bounds(positions, &BTreeMap::new()) else {
        return;
    };
    let delta = rect.center() - bounds.center();
    for pos in positions.values_mut() {
        *pos += delta;
    }
}

fn layout_iterations(node_count: usize) -> usize {
    match node_count {
        0..=40 => 70,
        41..=90 => 42,
        91..=150 => 24,
        _ => 12,
    }
}

fn layout_repulsion_stride(node_count: usize) -> usize {
    match node_count {
        0..=120 => 1,
        121..=190 => 2,
        _ => 3,
    }
}

fn layout_fingerprint(model: &GraphModel, selection: &GraphSelection, rect: Rect) -> u64 {
    let mut hasher = DefaultHasher::new();
    selection.hash(&mut hasher);
    (rect.width().round() as i32).hash(&mut hasher);
    (rect.height().round() as i32).hash(&mut hasher);
    model.nodes.len().hash(&mut hasher);
    model.edges.len().hash(&mut hasher);
    for node in &model.nodes {
        node.id.hash(&mut hasher);
        node.kind.hash(&mut hasher);
    }
    for edge in &model.edges {
        edge.id.hash(&mut hasher);
        edge.from.hash(&mut hasher);
        edge.to.hash(&mut hasher);
        edge.weight.hash(&mut hasher);
    }
    hasher.finish()
}

fn selected_neighbor_nodes(model: &GraphModel, selection: &GraphSelection) -> HashSet<String> {
    let mut ids = HashSet::new();
    match selection {
        GraphSelection::Overview => {
            ids.extend(model.nodes.iter().map(|node| node.id.clone()));
        }
        GraphSelection::Node(id) => {
            ids.insert(id.clone());
            for edge in &model.edges {
                if &edge.from == id {
                    ids.insert(edge.to.clone());
                }
                if &edge.to == id {
                    ids.insert(edge.from.clone());
                }
            }
        }
        GraphSelection::Edge(id) => {
            if let Some(edge) = model.edges.iter().find(|edge| &edge.id == id) {
                ids.insert(edge.from.clone());
                ids.insert(edge.to.clone());
            }
        }
    }
    ids
}

fn selected_edges(model: &GraphModel, selection: &GraphSelection) -> HashSet<String> {
    match selection {
        GraphSelection::Overview => model.edges.iter().map(|edge| edge.id.clone()).collect(),
        GraphSelection::Node(id) => model
            .edges
            .iter()
            .filter(|edge| &edge.from == id || &edge.to == id)
            .map(|edge| edge.id.clone())
            .collect(),
        GraphSelection::Edge(id) => BTreeSet::from([id.clone()]).into_iter().collect(),
    }
}

fn explainable_paths(model: &GraphModel, start_id: &str, limit: usize) -> Vec<ExplainablePath> {
    if !model.nodes.iter().any(|node| node.id == start_id) {
        return Vec::new();
    }

    let node_ids = model
        .nodes
        .iter()
        .map(|node| node.id.as_str())
        .collect::<HashSet<_>>();
    let node_by_id = model
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect::<HashMap<_, _>>();
    let mut adjacency = BTreeMap::<String, Vec<(String, String)>>::new();
    let mut degree = HashMap::<String, usize>::new();
    for edge in &model.edges {
        if !node_ids.contains(edge.from.as_str()) || !node_ids.contains(edge.to.as_str()) {
            continue;
        }
        adjacency
            .entry(edge.from.clone())
            .or_default()
            .push((edge.to.clone(), edge.id.clone()));
        adjacency
            .entry(edge.to.clone())
            .or_default()
            .push((edge.from.clone(), edge.id.clone()));
        *degree.entry(edge.from.clone()).or_default() += edge.weight.max(1);
        *degree.entry(edge.to.clone()).or_default() += edge.weight.max(1);
    }
    for neighbors in adjacency.values_mut() {
        neighbors.sort_by(|left, right| {
            let left_priority = node_by_id
                .get(left.0.as_str())
                .map(|node| explain_target_priority(node.kind))
                .unwrap_or_default();
            let right_priority = node_by_id
                .get(right.0.as_str())
                .map(|node| explain_target_priority(node.kind))
                .unwrap_or_default();
            right_priority
                .cmp(&left_priority)
                .then_with(|| left.0.cmp(&right.0))
        });
    }

    let max_depth = if model.nodes.len() > 120 { 3 } else { 4 };
    let mut queue = VecDeque::from([start_id.to_string()]);
    let mut depth = HashMap::<String, usize>::from([(start_id.to_string(), 0)]);
    let mut parent = HashMap::<String, (String, String)>::new();
    while let Some(current) = queue.pop_front() {
        let current_depth = depth[&current];
        if current_depth >= max_depth {
            continue;
        }
        let Some(neighbors) = adjacency.get(&current) else {
            continue;
        };
        for (neighbor, edge_id) in neighbors {
            if depth.contains_key(neighbor) {
                continue;
            }
            depth.insert(neighbor.clone(), current_depth + 1);
            parent.insert(neighbor.clone(), (current.clone(), edge_id.clone()));
            queue.push_back(neighbor.clone());
        }
    }

    let start_kind = node_by_id
        .get(start_id)
        .map(|node| node.kind)
        .unwrap_or(GraphNodeKind::Entity);
    let mut paths = depth
        .iter()
        .filter(|(node_id, node_depth)| node_id.as_str() != start_id && **node_depth > 0)
        .filter_map(|(node_id, node_depth)| {
            let node = node_by_id.get(node_id.as_str())?;
            let node_degree = degree.get(node_id).copied().unwrap_or_default();
            let useful = *node_depth == 1 || node.kind != start_kind || node_degree >= 3;
            if !useful {
                return None;
            }
            let mut path = reconstruct_explainable_path(start_id, node_id, &parent)?;
            path.score = explain_target_priority(node.kind) * 100 + node_degree * 6
                - node_depth * 18
                + model
                    .edges
                    .iter()
                    .filter(|edge| path.edge_ids.contains(&edge.id))
                    .map(|edge| edge.weight.max(1))
                    .sum::<usize>();
            Some(path)
        })
        .collect::<Vec<_>>();

    paths.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.edge_ids.len().cmp(&right.edge_ids.len()))
            .then_with(|| {
                node_label(
                    model,
                    left.node_ids.last().map(String::as_str).unwrap_or(""),
                )
                .cmp(node_label(
                    model,
                    right.node_ids.last().map(String::as_str).unwrap_or(""),
                ))
            })
    });
    paths.truncate(limit.clamp(1, 12));
    paths
}

fn reconstruct_explainable_path(
    start_id: &str,
    target_id: &str,
    parent: &HashMap<String, (String, String)>,
) -> Option<ExplainablePath> {
    let mut node_ids = vec![target_id.to_string()];
    let mut edge_ids = Vec::new();
    let mut current = target_id.to_string();
    while current != start_id {
        let (previous, edge_id) = parent.get(&current)?;
        edge_ids.push(edge_id.clone());
        node_ids.push(previous.clone());
        current = previous.clone();
    }
    node_ids.reverse();
    edge_ids.reverse();
    Some(ExplainablePath {
        node_ids,
        edge_ids,
        score: 0,
    })
}

fn explain_target_priority(kind: GraphNodeKind) -> usize {
    match kind {
        GraphNodeKind::Cluster => 9,
        GraphNodeKind::CodeFile => 8,
        GraphNodeKind::Memory => 8,
        GraphNodeKind::CodeSymbol => 7,
        GraphNodeKind::Entity => 7,
        GraphNodeKind::Type => 6,
        GraphNodeKind::Fact => 4,
    }
}

fn format_explainable_path(model: &GraphModel, path: &ExplainablePath) -> String {
    if path.node_ids.is_empty() {
        return String::new();
    }
    let mut text = format!("{} шаг. ", path.edge_ids.len());
    text.push_str(&truncate_text(node_label(model, &path.node_ids[0]), 24));
    for index in 0..path.edge_ids.len() {
        let from_id = &path.node_ids[index];
        let to_id = &path.node_ids[index + 1];
        let Some(edge) = model
            .edges
            .iter()
            .find(|edge| edge.id == path.edge_ids[index])
        else {
            continue;
        };
        if &edge.from == from_id && &edge.to == to_id {
            text.push_str(&format!(
                " -{}→ {}",
                truncate_text(&edge.label, 18),
                truncate_text(node_label(model, to_id), 24)
            ));
        } else {
            text.push_str(&format!(
                " ←{}- {}",
                truncate_text(&edge.label, 18),
                truncate_text(node_label(model, to_id), 24)
            ));
        }
    }
    truncate_text(&text, 170)
}

fn transformed_graph_position(
    position: Option<&Pos2>,
    offset: Option<&Vec2>,
    rect: Rect,
    zoom: f32,
    pan: Vec2,
) -> Option<Pos2> {
    let position = *position?;
    let offset = offset.copied().unwrap_or(Vec2::ZERO);
    Some(rect.center() + ((position + offset) - rect.center()) * zoom + pan)
}

fn graph_screen_to_world(rect: Rect, position: Pos2, zoom: f32, pan: Vec2) -> Pos2 {
    rect.center() + (position - rect.center() - pan) / zoom.max(MIN_GRAPH_ZOOM)
}

fn graph_world_bounds(
    positions: &BTreeMap<String, Pos2>,
    offsets: &BTreeMap<String, Vec2>,
) -> Option<Rect> {
    let mut values = positions.iter();
    let (first_id, first_pos) = values.next()?;
    let first = *first_pos + offsets.get(first_id).copied().unwrap_or(Vec2::ZERO);
    let mut bounds = Rect::from_min_max(first, first);
    for (id, pos) in values {
        let pos = *pos + offsets.get(id).copied().unwrap_or(Vec2::ZERO);
        bounds = bounds.union(Rect::from_min_max(pos, pos));
    }
    Some(bounds.expand(190.0))
}

fn graph_world_bounds_for_model(
    model: &GraphModel,
    positions: &BTreeMap<String, Pos2>,
    offsets: &BTreeMap<String, Vec2>,
    zoom: f32,
) -> Option<Rect> {
    let mut bounds = None::<Rect>;
    for node in &model.nodes {
        let Some(pos) = positions.get(&node.id) else {
            continue;
        };
        let center = *pos + offsets.get(&node.id).copied().unwrap_or_default();
        let size = collision_node_size(node.kind, zoom)
            + Vec2::splat((NODE_COLLISION_PADDING * 2.0) / zoom.max(MIN_GRAPH_ZOOM));
        let rect = Rect::from_center_size(center, size);
        bounds = Some(match bounds {
            Some(bounds) => bounds.union(rect),
            None => rect,
        });
    }
    bounds.map(|bounds| bounds.expand(36.0 / zoom.max(MIN_GRAPH_ZOOM)))
}

fn edge_bounds(from: Pos2, to: Pos2) -> Rect {
    Rect::from_min_max(
        Pos2::new(from.x.min(to.x), from.y.min(to.y)),
        Pos2::new(from.x.max(to.x), from.y.max(to.y)),
    )
}

fn node_rect(pos: Pos2, kind: GraphNodeKind) -> Rect {
    Rect::from_center_size(pos, node_card_size(kind))
}

fn node_card_size(kind: GraphNodeKind) -> Vec2 {
    match kind {
        GraphNodeKind::Cluster => Vec2::new(215.0, 58.0),
        GraphNodeKind::Type => Vec2::new(185.0, 52.0),
        GraphNodeKind::CodeSymbol => Vec2::new(210.0, 52.0),
        GraphNodeKind::CodeFile => Vec2::new(235.0, 56.0),
        GraphNodeKind::Entity => Vec2::new(170.0, 46.0),
        GraphNodeKind::Memory => Vec2::new(235.0, 56.0),
        GraphNodeKind::Fact => Vec2::new(190.0, 44.0),
    }
}

fn parallel_edge_offsets(model: &GraphModel) -> BTreeMap<String, f32> {
    let mut grouped = BTreeMap::<(String, String), Vec<String>>::new();
    for edge in &model.edges {
        grouped
            .entry((edge.from.clone(), edge.to.clone()))
            .or_default()
            .push(edge.id.clone());
    }
    let mut offsets = BTreeMap::new();
    for ids in grouped.values() {
        if ids.len() <= 1 {
            continue;
        }
        let center = (ids.len() - 1) as f32 / 2.0;
        for (index, id) in ids.iter().enumerate() {
            offsets.insert(id.clone(), (index as f32 - center) * 10.0);
        }
    }
    offsets
}

fn offset_edge_points(from: Pos2, to: Pos2, offset: f32) -> (Pos2, Pos2) {
    if offset.abs() < f32::EPSILON {
        return (from, to);
    }
    let delta = to - from;
    let length = delta.length().max(1.0);
    let normal = Vec2::new(-delta.y / length, delta.x / length) * offset;
    (from + normal, to + normal)
}

fn draw_edge_arrow(painter: &egui::Painter, from: Pos2, to: Pos2, stroke: Stroke) {
    let delta = to - from;
    let length = delta.length();
    if length < 18.0 {
        return;
    }
    let direction = delta / length;
    let normal = Vec2::new(-direction.y, direction.x);
    let tip = from + delta * 0.82;
    let wing = 8.0;
    let spread = 4.5;
    painter.line_segment([tip, tip - direction * wing + normal * spread], stroke);
    painter.line_segment([tip, tip - direction * wing - normal * spread], stroke);
}

fn edge_kind_color(kind: GraphEdgeKind) -> Color32 {
    match kind {
        GraphEdgeKind::Aggregate => Color32::from_rgb(94, 88, 116),
        GraphEdgeKind::CodeRelation => Color32::from_rgb(77, 112, 156),
        GraphEdgeKind::Explicit => Color32::from_rgb(82, 135, 103),
        GraphEdgeKind::Fact => Color32::from_rgb(132, 91, 150),
        GraphEdgeKind::Evidence => Color32::from_rgb(151, 111, 57),
        GraphEdgeKind::Derived => Color32::from_rgb(92, 92, 108),
    }
}

fn edge_kind_focus_color(kind: GraphEdgeKind) -> Color32 {
    match kind {
        GraphEdgeKind::Aggregate => Color32::from_rgb(139, 127, 180),
        GraphEdgeKind::CodeRelation => Color32::from_rgb(108, 153, 210),
        GraphEdgeKind::Explicit => Color32::from_rgb(111, 184, 139),
        GraphEdgeKind::Fact => Color32::from_rgb(177, 124, 201),
        GraphEdgeKind::Evidence => Color32::from_rgb(207, 151, 76),
        GraphEdgeKind::Derived => Color32::from_rgb(140, 140, 158),
    }
}

fn node_hover_text(node: &GraphNode, view_mode: GraphViewMode) -> String {
    let action = if node.id.starts_with("overview:") {
        "\nДвойной клик: открыть область"
    } else if view_mode == GraphViewMode::Clusters && node.kind == GraphNodeKind::Cluster {
        "\nДвойной клик: открыть кластер"
    } else {
        ""
    };
    format!("{}\n{}{}", node.label, node.detail, action)
}

fn draw_graph_legend(painter: &egui::Painter, rect: Rect) {
    let node_entries = [
        (GraphNodeKind::Cluster, "Кластеры"),
        (GraphNodeKind::Memory, "Память"),
        (GraphNodeKind::Entity, "Сущности"),
        (GraphNodeKind::Fact, "Факты"),
        (GraphNodeKind::CodeSymbol, "Символы"),
        (GraphNodeKind::CodeFile, "Файлы"),
        (GraphNodeKind::Type, "Типы"),
    ];
    let edge_entries = [
        GraphEdgeKind::Aggregate,
        GraphEdgeKind::Explicit,
        GraphEdgeKind::CodeRelation,
        GraphEdgeKind::Fact,
        GraphEdgeKind::Evidence,
        GraphEdgeKind::Derived,
    ];
    let legend_rect = Rect::from_min_size(
        Pos2::new(rect.left() + 14.0, rect.bottom() - 152.0),
        Vec2::new(308.0, 138.0),
    );
    painter.rect_filled(
        legend_rect,
        8.0,
        Color32::from_rgba_unmultiplied(30, 30, 38, 220),
    );
    painter.rect_stroke(
        legend_rect,
        8.0,
        Stroke::new(1.0, Color32::from_rgb(58, 55, 72)),
        egui::StrokeKind::Middle,
    );
    painter.text(
        legend_rect.left_top() + Vec2::new(10.0, 10.0),
        Align2::LEFT_TOP,
        "Легенда",
        FontId::proportional(12.0),
        Color32::from_rgb(226, 222, 238),
    );
    painter.text(
        legend_rect.left_top() + Vec2::new(166.0, 10.0),
        Align2::LEFT_TOP,
        "Связи",
        FontId::proportional(12.0),
        Color32::from_rgb(226, 222, 238),
    );
    for (index, (kind, label)) in node_entries.iter().enumerate() {
        let y = legend_rect.top() + 32.0 + index as f32 * 14.5;
        let swatch = Rect::from_min_size(Pos2::new(legend_rect.left() + 10.0, y), Vec2::splat(9.0));
        painter.rect_filled(swatch, 2.0, node_kind_color(*kind));
        painter.rect_stroke(
            swatch,
            2.0,
            Stroke::new(1.0, Color32::from_rgb(83, 80, 101)),
            egui::StrokeKind::Middle,
        );
        painter.text(
            Pos2::new(legend_rect.left() + 26.0, y + 4.5),
            Align2::LEFT_CENTER,
            *label,
            FontId::proportional(10.5),
            Color32::from_rgb(190, 186, 205),
        );
    }
    for (index, kind) in edge_entries.iter().enumerate() {
        let y = legend_rect.top() + 34.0 + index as f32 * 15.0;
        let from = Pos2::new(legend_rect.left() + 166.0, y);
        let to = Pos2::new(legend_rect.left() + 188.0, y);
        let stroke = Stroke::new(1.5, edge_kind_focus_color(*kind));
        painter.line_segment([from, to], stroke);
        draw_edge_arrow(painter, from, to, stroke);
        painter.text(
            Pos2::new(legend_rect.left() + 198.0, y),
            Align2::LEFT_CENTER,
            kind.label(),
            FontId::proportional(10.5),
            Color32::from_rgb(190, 186, 205),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_graph_minimap(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    rect: Rect,
    model: &GraphModel,
    positions: &BTreeMap<String, Pos2>,
    node_offsets: &BTreeMap<String, Vec2>,
    selection: &GraphSelection,
    zoom: f32,
    pan: Vec2,
) -> Option<Pos2> {
    let bounds = graph_world_bounds(positions, node_offsets)?;
    let minimap_size = Vec2::new(178.0, 126.0);
    let minimap_rect = Rect::from_min_size(
        Pos2::new(rect.right() - minimap_size.x - 14.0, rect.top() + 14.0),
        minimap_size,
    );
    let map_rect = Rect::from_min_max(
        minimap_rect.left_top() + Vec2::new(9.0, 28.0),
        minimap_rect.right_bottom() - Vec2::new(9.0, 9.0),
    );

    painter.rect_filled(
        minimap_rect,
        8.0,
        Color32::from_rgba_unmultiplied(30, 30, 38, 226),
    );
    painter.rect_stroke(
        minimap_rect,
        8.0,
        Stroke::new(1.0, Color32::from_rgb(58, 55, 72)),
        egui::StrokeKind::Middle,
    );
    painter.text(
        minimap_rect.left_top() + Vec2::new(10.0, 10.0),
        Align2::LEFT_CENTER,
        "Миникарта",
        FontId::proportional(11.5),
        Color32::from_rgb(226, 222, 238),
    );

    let world_to_minimap = |world: Pos2| -> Pos2 {
        let x = if bounds.width() <= 1.0 {
            0.5
        } else {
            (world.x - bounds.left()) / bounds.width()
        };
        let y = if bounds.height() <= 1.0 {
            0.5
        } else {
            (world.y - bounds.top()) / bounds.height()
        };
        Pos2::new(
            map_rect.left() + x.clamp(0.0, 1.0) * map_rect.width(),
            map_rect.top() + y.clamp(0.0, 1.0) * map_rect.height(),
        )
    };
    let minimap_to_world = |pos: Pos2| -> Pos2 {
        let x = ((pos.x - map_rect.left()) / map_rect.width()).clamp(0.0, 1.0);
        let y = ((pos.y - map_rect.top()) / map_rect.height()).clamp(0.0, 1.0);
        Pos2::new(
            bounds.left() + x * bounds.width(),
            bounds.top() + y * bounds.height(),
        )
    };

    for edge in model.edges.iter().take(420) {
        let Some(from) = positions
            .get(&edge.from)
            .map(|pos| *pos + node_offsets.get(&edge.from).copied().unwrap_or(Vec2::ZERO))
        else {
            continue;
        };
        let Some(to) = positions
            .get(&edge.to)
            .map(|pos| *pos + node_offsets.get(&edge.to).copied().unwrap_or(Vec2::ZERO))
        else {
            continue;
        };
        painter.line_segment(
            [world_to_minimap(from), world_to_minimap(to)],
            Stroke::new(0.8, Color32::from_rgb(72, 70, 86)),
        );
    }

    for node in &model.nodes {
        let Some(pos) = positions
            .get(&node.id)
            .map(|pos| *pos + node_offsets.get(&node.id).copied().unwrap_or(Vec2::ZERO))
        else {
            continue;
        };
        let selected = matches!(selection, GraphSelection::Node(id) if id == &node.id);
        painter.circle_filled(
            world_to_minimap(pos),
            if selected { 3.4 } else { 2.3 },
            if selected {
                Color32::from_rgb(214, 198, 255)
            } else {
                node_kind_color(node.kind)
            },
        );
    }

    let view_a = graph_screen_to_world(rect, rect.left_top(), zoom, pan);
    let view_b = graph_screen_to_world(rect, rect.right_bottom(), zoom, pan);
    let view_world = Rect::from_min_max(
        Pos2::new(view_a.x.min(view_b.x), view_a.y.min(view_b.y)),
        Pos2::new(view_a.x.max(view_b.x), view_a.y.max(view_b.y)),
    );
    let view_min = world_to_minimap(view_world.left_top());
    let view_max = world_to_minimap(view_world.right_bottom());
    painter.rect_stroke(
        Rect::from_min_max(view_min, view_max),
        2.0,
        Stroke::new(1.2, Color32::from_rgb(184, 167, 255)),
        egui::StrokeKind::Middle,
    );

    let response = ui
        .interact(
            minimap_rect,
            ui.make_persistent_id("graph_minimap"),
            Sense::click_and_drag(),
        )
        .on_hover_cursor(egui::CursorIcon::Grab);
    if (response.clicked() || response.dragged())
        && let Some(pointer) = response.interact_pointer_pos()
        && map_rect.contains(pointer)
    {
        return Some(minimap_to_world(pointer));
    }
    None
}

fn draw_graph_node(
    painter: &egui::Painter,
    node: &GraphNode,
    rect: Rect,
    selected: bool,
    related: bool,
    hovered: bool,
    pinned: bool,
) {
    let fill = if selected {
        Color32::from_rgb(92, 75, 153)
    } else if hovered {
        Color32::from_rgb(50, 47, 63)
    } else if related {
        node_kind_color(node.kind)
    } else {
        Color32::from_rgb(33, 32, 41)
    };
    let stroke = if selected {
        Stroke::new(2.4, Color32::from_rgb(202, 191, 255))
    } else if hovered {
        Stroke::new(1.8, Color32::from_rgb(142, 124, 218))
    } else if related {
        Stroke::new(1.1, Color32::from_rgb(83, 80, 101))
    } else {
        Stroke::new(1.0, Color32::from_rgb(55, 52, 68))
    };
    painter.rect_filled(rect, 8.0, fill);
    painter.rect_stroke(rect, 8.0, stroke, egui::StrokeKind::Middle);
    if pinned {
        painter.circle_filled(
            rect.right_top() + Vec2::new(-9.0, 9.0),
            3.5,
            Color32::from_rgb(227, 157, 74),
        );
    }
    painter.text(
        rect.left_top() + Vec2::new(10.0, 9.0),
        Align2::LEFT_TOP,
        truncate_text(&node.label, 34),
        FontId::proportional(12.5),
        if selected {
            Color32::WHITE
        } else {
            Color32::from_rgb(226, 222, 238)
        },
    );
    painter.text(
        rect.left_bottom() + Vec2::new(10.0, -10.0),
        Align2::LEFT_BOTTOM,
        truncate_text(&node.detail, 28),
        FontId::proportional(10.5),
        if selected {
            Color32::from_rgb(231, 226, 255)
        } else {
            Color32::from_rgb(151, 148, 164)
        },
    );
}

fn node_kind_color(kind: GraphNodeKind) -> Color32 {
    match kind {
        GraphNodeKind::Cluster => Color32::from_rgb(55, 48, 75),
        GraphNodeKind::Type => Color32::from_rgb(48, 58, 49),
        GraphNodeKind::CodeSymbol => Color32::from_rgb(43, 52, 72),
        GraphNodeKind::CodeFile => Color32::from_rgb(39, 64, 72),
        GraphNodeKind::Entity => Color32::from_rgb(39, 66, 52),
        GraphNodeKind::Memory => Color32::from_rgb(45, 52, 65),
        GraphNodeKind::Fact => Color32::from_rgb(63, 48, 70),
    }
}

fn detail_row(ui: &mut egui::Ui, name: &str, value: &str, monospace: bool) {
    ui.label(RichText::new(name).strong());
    if monospace {
        ui.label(RichText::new(value).monospace());
    } else {
        ui.label(value);
    }
    ui.end_row();
}

fn retrieval_source_meter(ui: &mut egui::Ui, label: &str, count: usize, reference: usize) {
    let reference = reference.max(1);
    let value = (count as f32 / reference as f32).clamp(0.0, 1.0);
    ui.horizontal(|ui| {
        ui.add_sized([94.0, 18.0], egui::Label::new(label));
        ui.add(
            egui::ProgressBar::new(value)
                .desired_width(132.0)
                .text(count.to_string()),
        );
    });
}

fn format_percent(value: f32) -> String {
    format!("{:.0}%", value.clamp(0.0, 1.0) * 100.0)
}

fn status_badge(ui: &mut egui::Ui, status: &str) {
    ui.label(
        RichText::new(status_label(status))
            .small()
            .color(Color32::WHITE)
            .background_color(status_color(status)),
    );
}

fn status_color(status: &str) -> Color32 {
    match status {
        "active" => Color32::from_rgb(42, 120, 75),
        "pending" => Color32::from_rgb(154, 105, 20),
        "superseded" => Color32::from_rgb(88, 95, 105),
        "archived" => Color32::from_rgb(140, 55, 55),
        _ => Color32::from_rgb(75, 85, 99),
    }
}

fn status_label(status: &str) -> &str {
    match status {
        "active" => "активные",
        "pending" => "на проверке",
        "superseded" => "замененные",
        "archived" => "архив",
        other => other,
    }
}

fn display_relation(relation: &str) -> String {
    match relation {
        "uses" => "использует",
        "depends_on" => "зависит от",
        "enforces" => "закрепляет",
        "replaces" => "заменяет",
        "documents" => "документирует",
        "runs_before" => "перед",
        "runs_after" => "после",
        "belongs_to" => "часть",
        "fallbacks_to" => "резерв",
        "storage" => "хранение",
        "requires" => "требует",
        "produces" => "создает",
        "reads" => "читает",
        "writes" => "пишет",
        "calls" | "call" => "вызывает",
        "ra_call" => "вызывает",
        "ra_reference" => "ссылается",
        "declares_module" => "объявляет модуль",
        "cargo_package" => "пакет",
        "contains" => "содержит",
        "mentions_code" => "упоминает код",
        "relates_code" => "связано с кодом",
        other => return other.replace('_', " "),
    }
    .to_string()
}

fn node_label<'a>(model: &'a GraphModel, id: &'a str) -> &'a str {
    model
        .nodes
        .iter()
        .find(|node| node.id == id)
        .map(|node| node.label.as_str())
        .unwrap_or(id)
}

fn empty_graph() -> MemoryGraph {
    MemoryGraph {
        entities: Vec::new(),
        facts: Vec::new(),
        edges: Vec::new(),
    }
}

fn entity_node_id(id: &str) -> String {
    format!("entity:{id}")
}

fn type_node_id(entity_type: &str) -> String {
    format!("type:{entity_type}")
}

fn memory_cluster_node_id(cluster: &str) -> String {
    format!("memory-cluster:{cluster}")
}

fn memory_kind_cluster_node_id(kind: &str) -> String {
    format!("memory-kind:{kind}")
}

fn code_node_id(id: &str) -> String {
    format!("code:{id}")
}

fn code_kind_node_id(kind: &str) -> String {
    format!("code-kind:{kind}")
}

fn code_file_node_id(file_path: &str) -> String {
    format!("code-file:{file_path}")
}

fn code_dir_node_id(dir: &str) -> String {
    format!("code-dir:{dir}")
}

fn memory_node_id(id: &str) -> String {
    format!("memory:{id}")
}

fn fact_node_id(id: &str) -> String {
    format!("fact:{id}")
}

fn edge_edge_id(id: &str) -> String {
    format!("edge:{id}")
}

fn file_label(file_path: &str) -> String {
    Path::new(file_path)
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| file_path.to_string())
}

fn code_directory(file_path: &str) -> String {
    Path::new(file_path)
        .parent()
        .and_then(|parent| parent.to_str())
        .filter(|parent| !parent.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| "(root)".to_string())
}

fn memory_cluster_for_type(entity_type: &str) -> (&'static str, &'static str) {
    let value = entity_type.to_ascii_lowercase();
    if value.contains("project") || value.contains("domain") || value.contains("product") {
        ("project", "Проекты")
    } else if value.contains("code")
        || value.contains("module")
        || value.contains("file")
        || value.contains("symbol")
        || value.contains("api")
        || value.contains("command")
        || value.contains("tool")
    {
        ("code", "Код и инструменты")
    } else if value.contains("workflow")
        || value.contains("process")
        || value.contains("hook")
        || value.contains("automation")
        || value.contains("pipeline")
    {
        ("workflow", "Процессы")
    } else if value.contains("storage")
        || value.contains("database")
        || value.contains("table")
        || value.contains("schema")
        || value.contains("index")
    {
        ("storage", "Хранение")
    } else if value.contains("decision")
        || value.contains("rule")
        || value.contains("policy")
        || value.contains("constraint")
    {
        ("rules", "Решения и правила")
    } else {
        ("other", "Остальное")
    }
}

fn memory_cluster_label(cluster: &str) -> &'static str {
    match cluster {
        "project" => "Проекты",
        "code" => "Код и инструменты",
        "workflow" => "Процессы",
        "storage" => "Хранение",
        "rules" => "Решения и правила",
        _ => "Остальное",
    }
}

fn memory_search_text(memory: &Memory) -> String {
    let mut text = format!("{} {}", memory.kind, memory.body);
    if let Some(source) = &memory.source {
        text.push(' ');
        text.push_str(source);
    }
    for tag in &memory.tags {
        text.push(' ');
        text.push_str(tag);
    }
    text.to_ascii_lowercase()
}

fn entity_search_text(entity: &crate::store::MemoryEntity) -> String {
    let mut text = format!("{} {}", entity.entity_type, entity.name);
    if let Some(description) = &entity.description {
        text.push(' ');
        text.push_str(description);
    }
    for alias in &entity.aliases {
        text.push(' ');
        text.push_str(alias);
    }
    text.to_ascii_lowercase()
}

fn gui_settings_path(database_marker: &Path) -> PathBuf {
    database_marker
        .parent()
        .map(|parent| parent.join("gui-settings.json"))
        .unwrap_or_else(|| PathBuf::from("gui-settings.json"))
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn compact_graph_id(id: &str) -> String {
    let value = id
        .strip_prefix("overview:")
        .or_else(|| id.strip_prefix("memory-cluster:"))
        .or_else(|| id.strip_prefix("memory-kind:"))
        .or_else(|| id.strip_prefix("code-dir:"))
        .or_else(|| id.strip_prefix("code-file:"))
        .or_else(|| id.strip_prefix("code:"))
        .or_else(|| id.strip_prefix("entity:"))
        .or_else(|| id.strip_prefix("fact:"))
        .or_else(|| id.strip_prefix("edge:"))
        .unwrap_or(id);
    truncate_text(value, 36)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{CodeRelation, CodeSymbol, MemoryEdge, MemoryEntity, MemoryFact};

    #[test]
    fn type_model_aggregates_edges_by_entity_type() {
        let graph = MemoryGraph {
            entities: vec![
                entity("project-1", "project", "Dukememory"),
                entity("command-1", "command", "dukememory_prepare"),
                entity("command-2", "command", "dukememory_code_search"),
            ],
            facts: Vec::<MemoryFact>::new(),
            edges: vec![
                edge(
                    "edge-1",
                    "project-1",
                    "Dukememory",
                    "command-1",
                    "dukememory_prepare",
                ),
                edge(
                    "edge-2",
                    "project-1",
                    "Dukememory",
                    "command-2",
                    "dukememory_code_search",
                ),
            ],
        };

        let model = build_type_model(&graph);
        assert_eq!(model.nodes.len(), 2);
        assert_eq!(model.edges.len(), 1);
        assert_eq!(model.edges[0].from, type_node_id("project"));
        assert_eq!(model.edges[0].to, type_node_id("command"));
        assert_eq!(model.edges[0].label, "использует ×2");
        assert_eq!(model.edges[0].weight, 2);
    }

    #[test]
    fn code_kind_model_aggregates_code_relations() {
        let symbols = vec![
            code_symbol("symbol-1", "main", "function"),
            code_symbol("symbol-2", "MemoryViewerApp", "struct"),
        ];
        let relations = vec![CodeRelation {
            id: "relation-1".to_string(),
            project_id: "project".to_string(),
            from_symbol_id: Some("symbol-1".to_string()),
            from_file_path: "src/main.rs".to_string(),
            relation_kind: "calls".to_string(),
            target_name: "MemoryViewerApp".to_string(),
            target_symbol_id: Some("symbol-2".to_string()),
        }];

        let model = build_code_kind_model(&symbols, &relations);
        assert_eq!(model.nodes.len(), 2);
        assert_eq!(model.edges.len(), 1);
        assert_eq!(model.edges[0].from, code_kind_node_id("function"));
        assert_eq!(model.edges[0].to, code_kind_node_id("struct"));
        assert_eq!(model.edges[0].label, "вызывает");
    }

    #[test]
    fn code_file_model_aggregates_code_relations() {
        let symbols = vec![
            code_symbol_in_file("symbol-1", "main", "function", "src/main.rs"),
            code_symbol_in_file("symbol-2", "run_memory_viewer", "function", "src/gui.rs"),
        ];
        let relations = vec![CodeRelation {
            id: "relation-1".to_string(),
            project_id: "project".to_string(),
            from_symbol_id: Some("symbol-1".to_string()),
            from_file_path: "src/main.rs".to_string(),
            relation_kind: "calls".to_string(),
            target_name: "run_memory_viewer".to_string(),
            target_symbol_id: Some("symbol-2".to_string()),
        }];

        let model = build_code_file_model(&symbols, &relations);
        assert_eq!(model.nodes.len(), 2);
        assert_eq!(model.edges.len(), 1);
        assert_eq!(model.edges[0].from, code_file_node_id("src/main.rs"));
        assert_eq!(model.edges[0].to, code_file_node_id("src/gui.rs"));
        assert_eq!(model.edges[0].label, "вызывает");
    }

    #[test]
    fn memory_cluster_model_groups_entity_types() {
        let graph = MemoryGraph {
            entities: vec![
                entity("project-1", "project", "Dukememory"),
                entity("command-1", "command", "dukememory_prepare"),
            ],
            facts: Vec::<MemoryFact>::new(),
            edges: vec![edge(
                "edge-1",
                "project-1",
                "Dukememory",
                "command-1",
                "dukememory_prepare",
            )],
        };

        let model = build_memory_cluster_model(&graph);
        assert!(model.nodes.iter().any(|node| node.label == "Проекты"));
        assert!(
            model
                .nodes
                .iter()
                .any(|node| node.label == "Код и инструменты")
        );
        assert_eq!(model.edges.len(), 1);
    }

    #[test]
    fn memory_cluster_drilldown_expands_to_concrete_entities() {
        let graph = MemoryGraph {
            entities: vec![
                entity("project-1", "project", "Dukememory"),
                entity("command-1", "command", "dukememory_prepare"),
            ],
            facts: Vec::<MemoryFact>::new(),
            edges: vec![edge(
                "edge-1",
                "project-1",
                "Dukememory",
                "command-1",
                "dukememory_prepare",
            )],
        };

        let model =
            build_memory_cluster_drilldown_model(&[], &graph, &memory_cluster_node_id("project"))
                .expect("cluster should expand");
        assert!(
            model
                .nodes
                .iter()
                .any(|node| node.id == entity_node_id("project-1"))
        );
        assert!(
            model
                .nodes
                .iter()
                .any(|node| node.id == entity_node_id("command-1"))
        );
        assert_eq!(model.edges.len(), 1);
    }

    #[test]
    fn code_cluster_drilldown_expands_to_files_in_directory() {
        let symbols = vec![
            code_symbol_in_file("symbol-1", "main", "function", "src/main.rs"),
            code_symbol_in_file("symbol-2", "run_memory_viewer", "function", "src/gui.rs"),
            code_symbol_in_file("symbol-3", "fixture", "function", "tests/fixture.rs"),
        ];
        let relations = vec![CodeRelation {
            id: "relation-1".to_string(),
            project_id: "project".to_string(),
            from_symbol_id: Some("symbol-1".to_string()),
            from_file_path: "src/main.rs".to_string(),
            relation_kind: "calls".to_string(),
            target_name: "run_memory_viewer".to_string(),
            target_symbol_id: Some("symbol-2".to_string()),
        }];

        let model =
            build_code_cluster_drilldown_model(&symbols, &relations, &code_dir_node_id("src"))
                .expect("cluster should expand");
        assert!(
            model
                .nodes
                .iter()
                .any(|node| node.id == code_file_node_id("src/main.rs"))
        );
        assert!(
            model
                .nodes
                .iter()
                .any(|node| node.id == code_file_node_id("src/gui.rs"))
        );
        assert!(
            !model
                .nodes
                .iter()
                .any(|node| node.id == code_file_node_id("tests/fixture.rs"))
        );
        assert_eq!(model.edges.len(), 1);
    }

    #[test]
    fn bridge_model_links_memory_to_referenced_code_file() {
        let memories = vec![memory("memory-1", "Decision mentions src/gui.rs")];
        let graph = MemoryGraph {
            entities: Vec::new(),
            facts: Vec::new(),
            edges: Vec::new(),
        };
        let symbols = vec![code_symbol_in_file(
            "symbol-1",
            "run_memory_viewer",
            "function",
            "src/gui.rs",
        )];

        let model = build_bridge_model(&memories, &graph, &symbols);
        assert!(
            model
                .nodes
                .iter()
                .any(|node| node.id == memory_node_id("memory-1"))
        );
        assert!(
            model
                .nodes
                .iter()
                .any(|node| node.id == code_file_node_id("src/gui.rs"))
        );
        assert_eq!(model.edges.len(), 1);
    }

    #[test]
    fn code_target_index_deduplicates_files_and_keeps_symbol_hits() {
        let symbols = vec![
            code_symbol_in_file("symbol-1", "run_memory_viewer", "function", "src/gui.rs"),
            code_symbol_in_file("symbol-2", "MemoryViewerApp", "struct", "src/gui.rs"),
        ];
        let index = CodeTargetIndex::new(&symbols);

        let targets =
            index.targets_for_text("decision mentions src/gui.rs and run_memory_viewer", 8);

        assert_eq!(
            targets
                .iter()
                .filter(|target| target.id == code_file_node_id("src/gui.rs"))
                .count(),
            1
        );
        assert!(
            targets
                .iter()
                .any(|target| target.id == code_node_id("symbol-1"))
        );
    }

    #[test]
    fn retrieval_quality_flags_single_source_query() {
        let memories = vec![memory("memory-1", "Decision without graph evidence")];
        let graph = empty_graph();
        let model = GraphModel {
            nodes: Vec::new(),
            edges: Vec::new(),
            fallback: false,
        };

        let quality = analyze_retrieval_quality("decision", &memories, &graph, &[], &[], &model);

        assert!(quality.query_active);
        assert_eq!(quality.source_count, 1);
        assert!(quality.warning_count >= 2);
        assert!(
            quality
                .recommendations()
                .contains(&"результат держится на одном источнике")
        );
        assert!(
            quality
                .recommendations()
                .contains(&"мало графовых оснований для найденной памяти")
        );
    }

    #[test]
    fn retrieval_quality_counts_bridge_matches_and_code_coverage() {
        let memories = vec![memory(
            "memory-1",
            "GUI decision mentions src/gui.rs and run_memory_viewer",
        )];
        let graph = MemoryGraph {
            entities: vec![entity("entity-1", "component", "GUI")],
            facts: vec![fact("fact-1", "entity-1", "memory-1")],
            edges: Vec::new(),
        };
        let symbols = vec![
            code_symbol_in_file("symbol-1", "run_memory_viewer", "function", "src/gui.rs"),
            code_symbol_in_file("symbol-2", "MemoryViewerApp", "struct", "src/gui.rs"),
        ];
        let relations = vec![CodeRelation {
            id: "relation-1".to_string(),
            project_id: "project".to_string(),
            from_symbol_id: Some("symbol-1".to_string()),
            from_file_path: "src/gui.rs".to_string(),
            relation_kind: "calls".to_string(),
            target_name: "MemoryViewerApp".to_string(),
            target_symbol_id: Some("symbol-2".to_string()),
        }];
        let model = build_bridge_model(&memories, &graph, &symbols);

        let quality =
            analyze_retrieval_quality("gui", &memories, &graph, &symbols, &relations, &model);

        assert_eq!(quality.memory_graph_coverage, 1.0);
        assert_eq!(quality.code_relation_coverage, 1.0);
        assert!(quality.bridge_edges >= 1);
        assert!(quality.source_count >= 4);
        assert_eq!(quality.warning_count, 0);
    }

    #[test]
    fn project_overview_model_uses_aggregated_russian_nodes() {
        let memories = vec![memory("memory-1", "Project decision")];
        let graph = MemoryGraph {
            entities: vec![
                entity("project-1", "project", "Dukememory"),
                entity("command-1", "command", "dukememory_prepare"),
            ],
            facts: Vec::new(),
            edges: vec![edge(
                "edge-1",
                "project-1",
                "Dukememory",
                "command-1",
                "dukememory_prepare",
            )],
        };
        let symbols = vec![code_symbol_in_file(
            "symbol-1",
            "run_memory_viewer",
            "function",
            "src/gui.rs",
        )];

        let model = build_project_overview_model(&memories, &graph, &symbols, &[]);

        assert!(
            model
                .nodes
                .iter()
                .any(|node| node.id == "overview:project" && node.label == "Проект")
        );
        assert!(
            model
                .nodes
                .iter()
                .any(|node| node.id == "overview:memory" && node.label == "Память")
        );
        assert!(
            model
                .nodes
                .iter()
                .all(|node| node.label != "Memory" && node.label != "Graph")
        );
        assert!(
            model
                .edges
                .iter()
                .any(|edge| edge.from == "overview:project" && edge.to == "overview:memory")
        );
    }

    #[test]
    fn parallel_edge_offsets_separate_edges_with_same_endpoints() {
        let model = GraphModel {
            nodes: vec![
                GraphNode {
                    id: "a".to_string(),
                    label: "A".to_string(),
                    detail: String::new(),
                    kind: GraphNodeKind::Entity,
                },
                GraphNode {
                    id: "b".to_string(),
                    label: "B".to_string(),
                    detail: String::new(),
                    kind: GraphNodeKind::Entity,
                },
            ],
            edges: vec![
                GraphEdge {
                    id: "edge-1".to_string(),
                    from: "a".to_string(),
                    to: "b".to_string(),
                    label: "uses".to_string(),
                    kind: GraphEdgeKind::Explicit,
                    weight: 1,
                },
                GraphEdge {
                    id: "edge-2".to_string(),
                    from: "a".to_string(),
                    to: "b".to_string(),
                    label: "depends".to_string(),
                    kind: GraphEdgeKind::Explicit,
                    weight: 1,
                },
            ],
            fallback: false,
        };

        let offsets = parallel_edge_offsets(&model);

        assert_eq!(offsets.len(), 2);
        assert_ne!(offsets["edge-1"], offsets["edge-2"]);
        assert_eq!(offsets["edge-1"], -offsets["edge-2"]);
    }

    #[test]
    fn overview_navigation_targets_open_expected_views() {
        let facts = overview_navigation_target("overview:facts").expect("facts target");
        assert_eq!(facts.domain, GraphDomain::Memory);
        assert_eq!(facts.view_mode, GraphViewMode::Entities);
        assert!(facts.show_facts);

        let files = overview_navigation_target("overview:files").expect("files target");
        assert_eq!(files.domain, GraphDomain::Code);
        assert_eq!(files.view_mode, GraphViewMode::Files);
        assert!(!files.show_facts);

        assert!(overview_navigation_target("entity:unknown").is_none());
    }

    #[test]
    fn project_overview_layout_uses_stable_sides() {
        let memories = vec![memory("memory-1", "Project decision")];
        let graph = MemoryGraph {
            entities: vec![entity("project-1", "project", "Dukememory")],
            facts: Vec::<MemoryFact>::new(),
            edges: Vec::new(),
        };
        let symbols = vec![code_symbol_in_file(
            "symbol-1",
            "run_memory_viewer",
            "function",
            "src/gui.rs",
        )];
        let model = build_project_overview_model(&memories, &graph, &symbols, &[]);
        let rect = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(1000.0, 700.0));

        let positions = layout_graph(&model, rect, &GraphSelection::Overview);

        let project = positions["overview:project"];
        let memory = positions["overview:memory"];
        let code = positions["overview:code"];
        assert_eq!(project, rect.center());
        assert!(memory.x < project.x);
        assert!(code.x > project.x);
    }

    #[test]
    fn graph_layout_separates_memory_cards() {
        let nodes = (0..10)
            .map(|index| GraphNode {
                id: format!("memory:{index}"),
                label: format!("Memory {index}"),
                detail: "decision".to_string(),
                kind: GraphNodeKind::Memory,
            })
            .collect::<Vec<_>>();
        let edges = (0..9)
            .map(|index| GraphEdge {
                id: format!("edge:{index}"),
                from: format!("memory:{index}"),
                to: format!("memory:{}", index + 1),
                label: "связано".to_string(),
                kind: GraphEdgeKind::Derived,
                weight: 1,
            })
            .collect::<Vec<_>>();
        let model = GraphModel {
            nodes,
            edges,
            fallback: false,
        };
        let rect = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(900.0, 620.0));

        let positions = layout_graph(&model, rect, &GraphSelection::Overview);

        assert_no_node_overlaps(&model, &positions);
    }

    #[test]
    fn radial_graph_layout_separates_code_cards() {
        let nodes = (0..80)
            .map(|index| GraphNode {
                id: format!("code:symbol-{index}"),
                label: format!("symbol_{index}"),
                detail: "function · pomme-client/src/lib.rs".to_string(),
                kind: GraphNodeKind::CodeSymbol,
            })
            .collect::<Vec<_>>();
        let edges = (0..79)
            .map(|index| GraphEdge {
                id: format!("code-edge:{index}"),
                from: format!("code:symbol-{index}"),
                to: format!("code:symbol-{}", index + 1),
                label: "вызывает".to_string(),
                kind: GraphEdgeKind::CodeRelation,
                weight: 1,
            })
            .collect::<Vec<_>>();
        let model = GraphModel {
            nodes,
            edges,
            fallback: false,
        };
        let rect = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(760.0, 790.0));

        let positions = layout_graph(&model, rect, &GraphSelection::Overview);

        assert_no_node_overlaps(&model, &positions);
        let center = rect.center();
        let quadrants = positions
            .values()
            .map(|pos| (pos.x >= center.x, pos.y >= center.y))
            .collect::<HashSet<_>>();
        let max_radius = positions
            .values()
            .map(|pos| (*pos - center).length())
            .fold(0.0, f32::max);
        assert!(quadrants.len() >= 4);
        assert!(max_radius > rect.width().min(rect.height()) * 0.55);
    }

    #[test]
    fn saved_node_offsets_are_repelled_before_render() {
        let nodes = (0..18)
            .map(|index| GraphNode {
                id: format!("code:symbol-{index}"),
                label: format!("symbol_{index}"),
                detail: "function · src/gui.rs".to_string(),
                kind: GraphNodeKind::CodeSymbol,
            })
            .collect::<Vec<_>>();
        let model = GraphModel {
            nodes,
            edges: Vec::new(),
            fallback: false,
        };
        let mut positions = BTreeMap::new();
        for node in &model.nodes {
            positions.insert(node.id.clone(), Pos2::new(400.0, 300.0));
        }
        let mut offsets = BTreeMap::new();
        offsets.insert("code:symbol-0".to_string(), Vec2::ZERO);
        let pinned = BTreeSet::from(["code:symbol-0".to_string()]);

        resolve_node_offset_overlaps(
            &model,
            &positions,
            &mut offsets,
            MIN_GRAPH_ZOOM,
            Some("code:symbol-0"),
            &pinned,
        );

        assert_eq!(offsets.get("code:symbol-0").copied(), Some(Vec2::ZERO));
        assert_no_node_overlaps_at_zoom(&model, &positions, &offsets, MIN_GRAPH_ZOOM);
    }

    #[test]
    fn explainable_paths_find_short_memory_to_code_path() {
        let model = GraphModel {
            nodes: vec![
                GraphNode {
                    id: "memory:decision".to_string(),
                    label: "Решение".to_string(),
                    detail: "память".to_string(),
                    kind: GraphNodeKind::Memory,
                },
                GraphNode {
                    id: "entity:gui".to_string(),
                    label: "GUI".to_string(),
                    detail: "component".to_string(),
                    kind: GraphNodeKind::Entity,
                },
                GraphNode {
                    id: "code-file:src/gui.rs".to_string(),
                    label: "src/gui.rs".to_string(),
                    detail: "12 символов".to_string(),
                    kind: GraphNodeKind::CodeFile,
                },
            ],
            edges: vec![
                GraphEdge {
                    id: "edge:memory-gui".to_string(),
                    from: "memory:decision".to_string(),
                    to: "entity:gui".to_string(),
                    label: "описывает".to_string(),
                    kind: GraphEdgeKind::Evidence,
                    weight: 1,
                },
                GraphEdge {
                    id: "edge:gui-file".to_string(),
                    from: "entity:gui".to_string(),
                    to: "code-file:src/gui.rs".to_string(),
                    label: "реализован в".to_string(),
                    kind: GraphEdgeKind::Derived,
                    weight: 1,
                },
            ],
            fallback: false,
        };

        let paths = explainable_paths(&model, "memory:decision", 6);
        let path = paths
            .iter()
            .find(|path| {
                path.node_ids
                    .last()
                    .is_some_and(|id| id == "code-file:src/gui.rs")
            })
            .expect("code file path should be present");

        assert_eq!(
            path.node_ids,
            vec![
                "memory:decision".to_string(),
                "entity:gui".to_string(),
                "code-file:src/gui.rs".to_string()
            ]
        );
        let formatted = format_explainable_path(&model, path);
        assert!(formatted.contains("Решение -описывает→ GUI"));
        assert!(formatted.contains("GUI -реализован в→ src/gui.rs"));
    }

    #[test]
    fn explainable_path_format_preserves_reverse_direction() {
        let model = GraphModel {
            nodes: vec![
                GraphNode {
                    id: "entity:gui".to_string(),
                    label: "GUI".to_string(),
                    detail: "component".to_string(),
                    kind: GraphNodeKind::Entity,
                },
                GraphNode {
                    id: "code-file:src/gui.rs".to_string(),
                    label: "src/gui.rs".to_string(),
                    detail: "12 символов".to_string(),
                    kind: GraphNodeKind::CodeFile,
                },
            ],
            edges: vec![GraphEdge {
                id: "edge:gui-file".to_string(),
                from: "entity:gui".to_string(),
                to: "code-file:src/gui.rs".to_string(),
                label: "реализован в".to_string(),
                kind: GraphEdgeKind::Derived,
                weight: 1,
            }],
            fallback: false,
        };
        let path = ExplainablePath {
            node_ids: vec!["code-file:src/gui.rs".to_string(), "entity:gui".to_string()],
            edge_ids: vec!["edge:gui-file".to_string()],
            score: 1,
        };

        let formatted = format_explainable_path(&model, &path);

        assert!(formatted.contains("src/gui.rs ←реализован в- GUI"));
    }

    fn entity(id: &str, entity_type: &str, name: &str) -> MemoryEntity {
        MemoryEntity {
            id: id.to_string(),
            project_id: "project".to_string(),
            entity_type: entity_type.to_string(),
            name: name.to_string(),
            aliases: Vec::new(),
            description: None,
            created_at: "now".to_string(),
            updated_at: "now".to_string(),
        }
    }

    fn edge(
        id: &str,
        from_entity_id: &str,
        from_entity_name: &str,
        to_entity_id: &str,
        to_entity_name: &str,
    ) -> MemoryEdge {
        MemoryEdge {
            id: id.to_string(),
            project_id: "project".to_string(),
            from_entity_id: from_entity_id.to_string(),
            from_entity_name: from_entity_name.to_string(),
            to_entity_id: to_entity_id.to_string(),
            to_entity_name: to_entity_name.to_string(),
            relation_type: "uses".to_string(),
            memory_id: None,
            episode_id: None,
            confidence: 0.9,
            valid_from: None,
            valid_to: None,
            invalidated_by: None,
            observed_at: "now".to_string(),
        }
    }

    fn fact(id: &str, entity_id: &str, memory_id: &str) -> MemoryFact {
        MemoryFact {
            id: id.to_string(),
            project_id: "project".to_string(),
            entity_id: Some(entity_id.to_string()),
            memory_id: Some(memory_id.to_string()),
            episode_id: None,
            predicate: "documents".to_string(),
            value: "GUI".to_string(),
            confidence: 0.9,
            valid_from: None,
            valid_to: None,
            invalidated_by: None,
            observed_at: "now".to_string(),
        }
    }

    fn code_symbol(id: &str, name: &str, kind: &str) -> CodeSymbol {
        code_symbol_in_file(id, name, kind, "src/main.rs")
    }

    fn memory(id: &str, body: &str) -> Memory {
        Memory {
            id: id.to_string(),
            project_id: "project".to_string(),
            scope: "project".to_string(),
            memory_tier: "archival".to_string(),
            kind: "decision".to_string(),
            body: body.to_string(),
            tags: Vec::new(),
            source: None,
            status: "active".to_string(),
            importance: 0.7,
            confidence: 0.8,
            superseded_by: None,
            status_reason: None,
            score: None,
            quality_score: 0.0,
            usage_count: 0,
            last_used_at: None,
            contradiction_risk: 0.0,
            created_at: "now".to_string(),
            updated_at: "now".to_string(),
        }
    }

    fn code_symbol_in_file(id: &str, name: &str, kind: &str, file_path: &str) -> CodeSymbol {
        CodeSymbol {
            id: id.to_string(),
            project_id: "project".to_string(),
            file_path: file_path.to_string(),
            language: "rust".to_string(),
            name: name.to_string(),
            kind: kind.to_string(),
            signature: format!("fn {name}()"),
            body: String::new(),
            start_line: 1,
            end_line: 2,
            parent_id: None,
        }
    }

    fn assert_no_node_overlaps(model: &GraphModel, positions: &BTreeMap<String, Pos2>) {
        assert_no_node_overlaps_at_zoom(model, positions, &BTreeMap::new(), MIN_GRAPH_ZOOM);
    }

    fn assert_no_node_overlaps_at_zoom(
        model: &GraphModel,
        positions: &BTreeMap<String, Pos2>,
        offsets: &BTreeMap<String, Vec2>,
        zoom: f32,
    ) {
        for i in 0..model.nodes.len() {
            for j in (i + 1)..model.nodes.len() {
                let a = &model.nodes[i];
                let b = &model.nodes[j];
                let a_pos = positions[&a.id] + offsets.get(&a.id).copied().unwrap_or_default();
                let b_pos = positions[&b.id] + offsets.get(&b.id).copied().unwrap_or_default();
                let a_rect = node_rect(Pos2::new(a_pos.x * zoom, a_pos.y * zoom), a.kind);
                let b_rect = node_rect(Pos2::new(b_pos.x * zoom, b_pos.y * zoom), b.kind);
                assert!(
                    !a_rect.expand(1.0).intersects(b_rect.expand(1.0)),
                    "{} overlaps {}",
                    a.id,
                    b.id
                );
            }
        }
    }
}
