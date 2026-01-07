use std::collections::BTreeMap;

use criterion::{
    black_box,
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
};
use serde::{
    Deserialize,
    Serialize,
};
use serde_json::{
    json,
    Value,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct LlmFixture {
    request: ChatRequest,
    response: ChatResponse,
    structured_output: StructuredOutput,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ChatRequest {
    model: String,
    temperature: f32,
    max_tokens: u32,
    response_format: ResponseFormat,
    tools: Vec<ToolDefinition>,
    messages: Vec<Message>,
    metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ToolDefinition {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
enum ResponseFormat {
    JsonSchema { json_schema: JsonSchemaSpec },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct JsonSchemaSpec {
    name: String,
    strict: bool,
    schema: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Message {
    role: Role,
    content: String,
    name: Option<String>,
    tool_calls: Vec<ToolCall>,
    tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ToolCall {
    id: String,
    function: ToolFunctionCall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ToolFunctionCall {
    name: String,
    arguments: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ChatResponse {
    id: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Choice {
    index: u32,
    finish_reason: String,
    message: Message,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct StructuredOutput {
    answer: String,
    citations: Vec<Citation>,
    entities: Vec<Entity>,
    action_items: Vec<ActionItem>,
    confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Citation {
    source_id: String,
    chunk_index: u32,
    quote: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Entity {
    id: String,
    entity_type: String,
    name: String,
    attributes: BTreeMap<String, Value>,
    mentions: Vec<Mention>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Mention {
    source_id: String,
    span: Span,
    confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Span {
    start: u32,
    end: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ActionItem {
    title: String,
    owner: String,
    due_date: String,
    priority: String,
    tags: Vec<String>,
    context: Value,
}

#[derive(Clone, Copy, Debug)]
enum FixtureSize {
    Small,
    Medium,
    Large,
}

fn make_words(prefix: &str, words: usize) -> String {
    let mut out = String::new();
    for i in 0..words {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(prefix);
        out.push('_');
        out.push_str(&i.to_string());
    }
    out
}

fn make_tools(count: usize) -> Vec<ToolDefinition> {
    (0..count)
        .map(|i| ToolDefinition {
            name: format!("tool_{i}_extract"),
            description: format!(
                "Extract structured fields from a document chunk (tool index {i})."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "chunk_index": {"type": "integer"},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "constraints": {
                        "type": "object",
                        "properties": {
                            "date_range": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"}
                                }
                            },
                            "max_items": {"type": "integer"}
                        }
                    }
                },
                "required": ["source_id", "chunk_index", "fields"]
            }),
        })
        .collect()
}

fn make_messages(message_count: usize, avg_words: usize) -> Vec<Message> {
    let mut messages = Vec::with_capacity(message_count);

    messages.push(Message {
        role: Role::System,
        content: "You are a strict extraction engine. Respond only with JSON matching the schema."
            .to_string(),
        name: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
    });

    for i in 0..message_count {
        let role = if i % 5 == 0 {
            Role::Assistant
        } else {
            Role::User
        };

        let content = if matches!(role, Role::User) {
            format!(
                "source_id: doc_{i}\nchunk_index: {i}\ncontent: {}",
                make_words("token", avg_words + (i % 7))
            )
        } else {
            format!("Plan: extract entities, normalize dates, emit citations. Iteration={i}.")
        };

        let tool_calls = if matches!(role, Role::Assistant) && i % 10 == 0 {
            vec![ToolCall {
                id: format!("call_{i}"),
                function: ToolFunctionCall {
                    name: "tool_0_extract".to_string(),
                    arguments: json!({
                        "source_id": format!("doc_{i}"),
                        "chunk_index": i,
                        "fields": ["entities", "dates", "amounts"],
                        "constraints": {
                            "date_range": {"from": "2024-01-01", "to": "2026-12-31"},
                            "max_items": 64
                        }
                    }),
                },
            }]
        } else {
            Vec::new()
        };

        messages.push(Message {
            role,
            content,
            name: None,
            tool_calls,
            tool_call_id: None,
        });

        if i % 10 == 0 {
            messages.push(Message {
                role: Role::Tool,
                content: json!({
                    "source_id": format!("doc_{i}"),
                    "chunk_index": i,
                    "entities": [
                        {"type": "person", "name": format!("Person_{i}")},
                        {"type": "org", "name": format!("Org_{i}")}
                    ],
                    "dates": ["2025-01-01"],
                    "amounts": [{"currency": "USD", "value": i * 10}]
                })
                .to_string(),
                name: None,
                tool_calls: Vec::new(),
                tool_call_id: Some(format!("call_{i}")),
            });
        }
    }

    messages
}

fn make_structured_output(entity_count: usize, citation_count: usize) -> StructuredOutput {
    let citations = (0..citation_count)
        .map(|i| Citation {
            source_id: format!("doc_{}", i % 128),
            chunk_index: (i % 64) as u32,
            quote: make_words("quote", 12 + (i % 5)),
        })
        .collect();

    let entities = (0..entity_count)
        .map(|i| {
            let mut attributes = BTreeMap::new();
            attributes.insert("category".to_string(), Value::String("demo".to_string()));
            attributes.insert(
                "signals".to_string(),
                json!([
                    {"kind": "score", "value": (i % 100) as f64 / 100.0},
                    {"kind": "rank", "value": (i % 10)}
                ]),
            );
            attributes.insert(
                "metadata".to_string(),
                json!({
                    "source": format!("doc_{}", i % 64),
                    "chunk_index": i % 32,
                    "normalized": true
                }),
            );

            Entity {
                id: format!("ent_{i}"),
                entity_type: if i % 3 == 0 {
                    "person".to_string()
                } else if i % 3 == 1 {
                    "org".to_string()
                } else {
                    "location".to_string()
                },
                name: format!("Entity_{i}"),
                attributes,
                mentions: (0..(1 + (i % 3)))
                    .map(|j| Mention {
                        source_id: format!("doc_{}", (i + j) % 128),
                        span: Span {
                            start: ((i * 7 + j) % 1000) as u32,
                            end: ((i * 7 + j) % 1000 + 12) as u32,
                        },
                        confidence: 0.7 + (j as f32 * 0.1),
                    })
                    .collect(),
            }
        })
        .collect();

    let action_items = (0..(entity_count / 8).max(1))
        .map(|i| ActionItem {
            title: format!("Follow up on entity ent_{}", i * 3),
            owner: if i % 2 == 0 { "alice" } else { "bob" }.to_string(),
            due_date: format!("2026-{:02}-{:02}", (i % 12) + 1, (i % 28) + 1),
            priority: if i % 3 == 0 {
                "high"
            } else if i % 3 == 1 {
                "medium"
            } else {
                "low"
            }
            .to_string(),
            tags: vec![
                "llm".to_string(),
                "extraction".to_string(),
                format!("tag{i}"),
            ],
            context: json!({
                "entity_id": format!("ent_{}", i * 3),
                "rationale": make_words("rationale", 18 + (i % 4)),
                "references": [format!("doc_{}", i % 16), format!("doc_{}", (i + 1) % 16)]
            }),
        })
        .collect();

    StructuredOutput {
        answer: make_words("answer", 48),
        citations,
        entities,
        action_items,
        confidence: 0.86,
    }
}

fn make_fixture(size: FixtureSize) -> LlmFixture {
    let (message_count, avg_words, tool_count, entity_count, citation_count) = match size {
        FixtureSize::Small => (8, 40, 1, 24, 12),
        FixtureSize::Medium => (32, 80, 3, 160, 48),
        FixtureSize::Large => (96, 120, 8, 640, 160),
    };

    let mut metadata = BTreeMap::new();
    metadata.insert("tenant".to_string(), "acme".to_string());
    metadata.insert("request_id".to_string(), "req_demo_0001".to_string());
    metadata.insert("pipeline".to_string(), "structured_io".to_string());

    let request = ChatRequest {
        model: "gpt-4.1-mini".to_string(),
        temperature: 0.2,
        max_tokens: 2048,
        response_format: ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: "extraction_response".to_string(),
                strict: true,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "confidence": {"type": "number"},
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "entity_type": {"type": "string"},
                                    "name": {"type": "string"},
                                    "attributes": {"type": "object"}
                                },
                                "required": ["id", "entity_type", "name"]
                            }
                        }
                    },
                    "required": ["answer", "entities", "confidence"]
                }),
            },
        },
        tools: make_tools(tool_count),
        messages: make_messages(message_count, avg_words),
        metadata,
    };

    let structured_output = make_structured_output(entity_count, citation_count);

    let response = ChatResponse {
        id: "chatcmpl_demo_0001".to_string(),
        created: 1_725_000_000,
        model: request.model.clone(),
        choices: vec![Choice {
            index: 0,
            finish_reason: "stop".to_string(),
            message: Message {
                role: Role::Assistant,
                content: "{...structured output...}".to_string(),
                name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
        }],
        usage: Usage {
            prompt_tokens: 12_345,
            completion_tokens: 1_234,
            total_tokens: 13_579,
        },
    };

    LlmFixture {
        request,
        response,
        structured_output,
    }
}

fn bench_llm_fixtures(c: &mut Criterion) {
    let mut group = c.benchmark_group("serde_toon_vs_json_llm_fixtures");

    let fixtures = [
        ("small", make_fixture(FixtureSize::Small)),
        ("medium", make_fixture(FixtureSize::Medium)),
        ("large", make_fixture(FixtureSize::Large)),
    ];

    for (name, fixture) in fixtures {
        let json_str = serde_json::to_string(&fixture).expect("serialize json fixture");
        let toon_str = toon_format::to_string(&fixture).expect("serialize toon fixture");

        group.bench_with_input(
            BenchmarkId::new("json_encode", name),
            &fixture,
            |b, data| b.iter(|| black_box(serde_json::to_string(black_box(data)).unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("toon_encode", name),
            &fixture,
            |b, data| b.iter(|| black_box(toon_format::to_string(black_box(data)).unwrap())),
        );

        group.bench_with_input(BenchmarkId::new("json_decode", name), &json_str, |b, s| {
            b.iter(|| black_box(serde_json::from_str::<LlmFixture>(black_box(s)).unwrap()))
        });

        group.bench_with_input(BenchmarkId::new("toon_decode", name), &toon_str, |b, s| {
            b.iter(|| black_box(toon_format::from_str::<LlmFixture>(black_box(s)).unwrap()))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_llm_fixtures);
criterion_main!(benches);
