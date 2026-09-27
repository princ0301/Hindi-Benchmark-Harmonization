from src.benchmarks.loader import BenchmarkItem, register

CHATRAG_PROMPT_TEMPLATE = """निम्नलिखित संदर्भ अंशों के आधार पर बातचीत में अंतिम प्रश्न का उत्तर दें। उत्तर संक्षिप्त और केवल दिए गए संदर्भ पर आधारित होना चाहिए।

संदर्भ:
{context}

बातचीत:
{conversation}

सहायक:"""


def render_conversation(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        speaker = "उपयोगकर्ता" if msg["role"] == "user" else "सहायक"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n".join(lines)


@register("chatrag_hi")
def load_chatrag_hi() -> list[BenchmarkItem]:
    from datasets import load_dataset

    raw = load_dataset("nvidia/ChatRAG-Hi", "inscit", split="test")

    items = []
    dropped = 0
    for i, row in enumerate(raw):
        if not row["ground_truth_ctx"] or not row["answers"]:
            dropped += 1
            continue

        context = "\n\n".join(f"- {ctx['ctx']}" for ctx in row["ground_truth_ctx"])
        conversation = render_conversation(row["messages"])
        prompt = CHATRAG_PROMPT_TEMPLATE.format(context=context, conversation=conversation)

        items.append(
            BenchmarkItem(
                benchmark="chatrag_hi",
                item_id=f"chatrag_hi_{i:04d}",
                language="hi",
                task_type="generation",
                input={"topic": row["topic"]},
                gold=row["answers"],
                prompt_template=prompt,
            )
        )

    if dropped:
        print(f"chatrag_hi: dropped {dropped} item(s) with missing gold context or answers")

    return items