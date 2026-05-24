use std::collections::HashMap;
use tera::{Tera, Context};

pub type ConceptId = u64;

// ─── AST ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum AstNode {
    Sequence(Vec<AstNode>),
    FnCall { name: String, args: Vec<AstNode> },
    Var(String),
    Lit(String),
    Number(f64),
    BinOp { op: String, left: Box<AstNode>, right: Box<AstNode> },
    IfElse { cond: Box<AstNode>, then_body: Vec<AstNode>, else_body: Vec<AstNode> },
    Loop { var: String, iter: Box<AstNode>, body: Vec<AstNode> },
    Chain(Vec<AstNode>),
    Empty,
}

impl AstNode {
    pub fn render_text(&self, indent: usize) -> String {
        let pad = "  ".repeat(indent);
        match self {
            AstNode::Sequence(nodes) => {
                nodes.iter().map(|n| n.render_text(indent)).collect::<Vec<_>>().join("\n")
            }
            AstNode::FnCall { name, args } => {
                let args_str: Vec<String> = args.iter().map(|a| a.render_text(0)).collect();
                format!("{}({})", name, args_str.join(", "))
            }
            AstNode::Var(name) => name.clone(),
            AstNode::Lit(s) => format!("\"{}\"", s),
            AstNode::Number(n) => format!("{}", n),
            AstNode::BinOp { op, left, right } => {
                format!("{} {} {}", left.render_text(0), op, right.render_text(0))
            }
            AstNode::IfElse { cond, then_body, else_body } => {
                let mut s = format!("if ({}) {{\n", cond.render_text(0));
                for n in then_body { s.push_str(&format!("{}{}\n", pad, n.render_text(indent + 1))); }
                s.push_str(&format!("{}}}", pad));
                if !else_body.is_empty() {
                    s.push_str(" else {\n");
                    for n in else_body { s.push_str(&format!("{}{}\n", pad, n.render_text(indent + 1))); }
                    s.push_str(&format!("{}}}", pad));
                }
                s
            }
            AstNode::Loop { var, iter, body } => {
                let mut s = format!("for {} in {} {{\n", var, iter.render_text(0));
                for n in body { s.push_str(&format!("{}{}\n", pad, n.render_text(indent + 1))); }
                s.push_str(&format!("{}}}", pad));
                s
            }
            AstNode::Chain(nodes) => {
                nodes.iter().map(|n| n.render_text(indent)).collect::<Vec<_>>().join(" → ")
            }
            AstNode::Empty => String::new(),
        }
    }

    pub fn to_tera_context(&self) -> Context {
        let mut ctx = Context::new();
        match self {
            AstNode::Chain(nodes) => {
                let items: Vec<serde_json::Value> = nodes.iter().map(|n| {
                    serde_json::json!({"text": n.render_text(0), "type": n.node_type_label()})
                }).collect();
                ctx.insert("chain", &items);
                ctx
            }
            other => {
                let items = vec![
                    serde_json::json!({"text": other.render_text(0), "type": other.node_type_label()})
                ];
                ctx.insert("chain", &items);
                ctx
            }
        }
    }

    pub fn node_type_label(&self) -> &'static str {
        match self {
            AstNode::Sequence(_) => "sequence",
            AstNode::FnCall { .. } => "fn_call",
            AstNode::Var(_) => "variable",
            AstNode::Lit(_) => "literal",
            AstNode::Number(_) => "number",
            AstNode::BinOp { .. } => "binary_op",
            AstNode::IfElse { .. } => "if_else",
            AstNode::Loop { .. } => "loop",
            AstNode::Chain(_) => "chain",
            AstNode::Empty => "empty",
        }
    }
}

// ─── AST Generator ───────────────────────────────────────────

pub struct AstGenerator {
    known_fns: Vec<String>,
}

impl AstGenerator {
    pub fn new() -> Self {
        AstGenerator {
            known_fns: vec![
                "map".into(), "filter".into(), "reduce".into(), "fold".into(),
                "print".into(), "println".into(), "len".into(), "push".into(),
                "pop".into(), "get".into(), "set".into(), "sort".into(),
                "open".into(), "read".into(), "write".into(), "close".into(),
                "connect".into(), "send".into(), "recv".into(),
                "bind".into(), "link".into(), "chain".into(), "branch".into(),
            ],
        }
    }

    pub fn from_chain(&self, concepts: &[(ConceptId, &str, f32)]) -> AstNode {
        if concepts.is_empty() {
            return AstNode::Empty;
        }
        let mut nodes = Vec::new();
        for (i, &(_id, label, score)) in concepts.iter().enumerate() {
            if score < 0.01 { continue; }
            let cleaned: String = label.chars().filter(|c| c.is_alphanumeric() || *c == '_').collect();
            if cleaned.is_empty() { continue; }
            let node = if self.known_fns.contains(&cleaned) && i + 1 < concepts.len() {
                let rest: Vec<AstNode> = concepts[i + 1..].iter()
                    .take(3)
                    .map(|&(_id, l, _s)| {
                        let c: String = l.chars().filter(|c| c.is_alphanumeric() || *c == '_').collect();
                        AstNode::Var(c)
                    })
                    .collect();
                AstNode::FnCall { name: cleaned, args: rest }
            } else if cleaned == "if" || cleaned == "when" || cleaned == "branch" {
                let cond = if i + 1 < concepts.len() {
                    AstNode::Var(concepts[i + 1].1.chars().filter(|c| c.is_alphanumeric() || *c == '_').collect())
                } else {
                    AstNode::Lit("true".into())
                };
                let then_body = if i + 2 < concepts.len() {
                    vec![AstNode::Var(concepts[i + 2].1.chars().filter(|c| c.is_alphanumeric() || *c == '_').collect())]
                } else {
                    vec![]
                };
                AstNode::IfElse { cond: Box::new(cond), then_body, else_body: vec![] }
            } else if cleaned == "loop" || cleaned == "for" || cleaned == "each" {
                let var = "x".into();
                let iter = if i + 1 < concepts.len() {
                    AstNode::Var(concepts[i + 1].1.chars().filter(|c| c.is_alphanumeric() || *c == '_').collect())
                } else {
                    AstNode::Var("items".into())
                };
                AstNode::Loop { var, iter: Box::new(iter), body: vec![] }
            } else if i + 1 < concepts.len() && score > 0.5 {
                let next = concepts[i + 1];
                let next_clean: String = next.1.chars().filter(|c| c.is_alphanumeric() || *c == '_').collect();
                AstNode::BinOp {
                    op: "->".into(),
                    left: Box::new(AstNode::Var(cleaned)),
                    right: Box::new(AstNode::Var(next_clean)),
                }
            } else {
                AstNode::Var(cleaned)
            };
            nodes.push(node);
        }
        AstNode::Chain(nodes)
    }

    pub fn add_known_fn(&mut self, name: &str) {
        if !self.known_fns.contains(&name.to_string()) {
            self.known_fns.push(name.to_string());
        }
    }
}

impl Default for AstGenerator {
    fn default() -> Self { Self::new() }
}

// ─── Template Engine ─────────────────────────────────────────

pub struct TemplateEngine {
    tera: Tera,
}

impl TemplateEngine {
    pub fn new() -> Result<Self, tera::Error> {
        let mut tera = Tera::default();
        tera.add_raw_templates(vec![
            ("rust_fn", RUST_FN_TEMPLATE),
            ("rust_main", RUST_MAIN_TEMPLATE),
            ("python", PYTHON_TEMPLATE),
            ("bash", BASH_TEMPLATE),
            ("desc", DESC_TEMPLATE),
        ])?;
        Ok(TemplateEngine { tera })
    }

    pub fn render(&self, template: &str, ctx: &Context) -> Result<String, tera::Error> {
        self.tera.render(template, ctx)
    }

    pub fn render_chain(&self, ast: &AstNode, lang: &str) -> Result<String, tera::Error> {
        let ctx = ast.to_tera_context();
        let tmpl = match lang {
            "rust" => "rust_fn",
            "py" | "python" => "python",
            "sh" | "bash" => "bash",
            "desc" | "text" => "desc",
            _ => "desc",
        };
        self.render(tmpl, &ctx)
    }
}

const RUST_FN_TEMPLATE: &str = r#"
fn generated({{ chain | map(attribute="text") | join(sep=", ") }}) {
    println!("chain: {% for item in chain %}{{ item.text }} → {% endfor %}end");
}
"#;

const RUST_MAIN_TEMPLATE: &str = r#"
fn main() {
    let chain = vec![{% for item in chain %}"{{ item.text }}",{% endfor %}];
    println!("Chain: {:?}", chain);
}
"#;

const PYTHON_TEMPLATE: &str = r#"
def generated():
    chain = [{% for item in chain %}"{{ item.text }}",{% endfor %}]
    print(" → ".join(chain))
generated()
"#;

const BASH_TEMPLATE: &str = r#"#!/bin/bash
chain=({% for item in chain %}{{ item.text }} {% endfor %})
echo "Chain: ${chain[@]}"
"#;

const DESC_TEMPLATE: &str = r#"
Association chain: {% for item in chain %}{{ item.text }}{% if not loop.last %} → {% endif %}{% endfor %}
This chain contains {{ chain | length }} concepts.
"#;

// ─── WASM Sandbox ────────────────────────────────────────────

pub struct Sandbox {
    engine: wasmtime::Engine,
}

impl Sandbox {
    pub fn new() -> Result<Self, wasmtime::Error> {
        let engine = wasmtime::Engine::new(wasmtime::Config::new().debug_info(false))?;
        Ok(Sandbox { engine })
    }

    pub fn eval(&self, wasm_bytes: &[u8], func_name: &str, args: &[wasmtime::Val])
        -> Result<Option<wasmtime::Val>, wasmtime::Error>
    {
        let module = wasmtime::Module::new(&self.engine, wasm_bytes)?;
        let mut store = wasmtime::Store::new(&self.engine, ());
        let linker = wasmtime::Linker::new(&self.engine);
        let instance = linker.instantiate(&mut store, &module)?;
        let func = instance.get_func(&mut store, func_name)
            .ok_or_else(|| wasmtime::Error::msg(format!("func '{}' not found", func_name)))?;
        let mut results = vec![wasmtime::Val::I32(0)];
        func.call(&mut store, args, &mut results)?;
        Ok(results.into_iter().next())
    }

    pub fn compile_wat(&self, wat: &str) -> Result<Vec<u8>, wasmtime::Error> {
        let module = wasmtime::Module::new(&self.engine, wat)?;
        Ok(module.serialize()?.to_vec())
    }

    pub fn eval_wat(&self, wat: &str, func_name: &str, args: &[wasmtime::Val])
        -> Result<Option<wasmtime::Val>, wasmtime::Error>
    {
        let module = wasmtime::Module::new(&self.engine, wat)?;
        let mut store = wasmtime::Store::new(&self.engine, ());
        let linker = wasmtime::Linker::new(&self.engine);
        let instance = linker.instantiate(&mut store, &module)?;
        let func = instance.get_func(&mut store, func_name)
            .ok_or_else(|| wasmtime::Error::msg(format!("func '{}' not found", func_name)))?;
        let mut results = vec![wasmtime::Val::I32(0)];
        func.call(&mut store, args, &mut results)?;
        Ok(results.into_iter().next())
    }
}

impl Default for Sandbox {
    fn default() -> Self { Self::new().unwrap() }
}

// ─── Coder Pipeline ──────────────────────────────────────────

pub struct Coder {
    pub ast_gen: AstGenerator,
    pub templates: TemplateEngine,
    pub sandbox: Sandbox,
}

impl Coder {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Coder {
            ast_gen: AstGenerator::new(),
            templates: TemplateEngine::new()?,
            sandbox: Sandbox::new()?,
        })
    }

    pub fn chain_to_rust(&self, chain: &[(ConceptId, &str, f32)]) -> Result<String, tera::Error> {
        let ast = self.ast_gen.from_chain(chain);
        self.templates.render_chain(&ast, "rust")
    }

    pub fn chain_to_python(&self, chain: &[(ConceptId, &str, f32)]) -> Result<String, tera::Error> {
        let ast = self.ast_gen.from_chain(chain);
        self.templates.render_chain(&ast, "python")
    }

    pub fn chain_to_text(&self, chain: &[(ConceptId, &str, f32)]) -> Result<String, tera::Error> {
        let ast = self.ast_gen.from_chain(chain);
        self.templates.render_chain(&ast, "desc")
    }

    pub fn chain_to_ast_text(&self, chain: &[(ConceptId, &str, f32)]) -> String {
        let ast = self.ast_gen.from_chain(chain);
        ast.render_text(0)
    }

    pub fn run_wasm(&self, wasm: &[u8], func: &str, args: &[wasmtime::Val])
        -> Result<Option<wasmtime::Val>, wasmtime::Error>
    {
        self.sandbox.eval(wasm, func, args)
    }
}
