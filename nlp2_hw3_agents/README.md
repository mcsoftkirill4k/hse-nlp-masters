# NLP-2 HW3 — LangGraph airline assistant

Customer support бот на LangGraph: от zero-shot агента до multi-agent с safe/sensitive tools и handoff специалистам.

## Что сделано

- Part 1: assistant ↔ tools
- Part 2: `fetch_user_info` + `interrupt_before=["tools"]`
- Part 3: `route_tools` → safe / sensitive (interrupt только на запись)
- Part 4: specialists (flight / car / hotel / excursion) + primary router, `dialog_state`, `leave_skill`

## Запуск

```bash
cd nlp2_hw3_agents
# ключи: OPENROUTER_API_KEY, TAVILY_API_KEY (ноутбук спросит через getpass)
jupyter notebook nlp2_hw3_agents.ipynb
```

Нужны зависимости из первой ячейки ноутбука (langgraph, langchain-*, sqlite DB скачивается в setup). Без Tavily — убрать `TavilySearchResults` из списков tools.
