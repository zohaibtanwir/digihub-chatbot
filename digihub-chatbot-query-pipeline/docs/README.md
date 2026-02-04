# DigiHub Chatbot Documentation

This directory contains detailed feature documentation for the DigiHub Chatbot Query Pipeline.

## Feature Documentation

### [Conversational Context Engine](conversational-context-engine.md)

The **Conversational Context Engine (CCE)** is the core system that enables natural multi-turn conversations by maintaining context across chat sessions.

**Key Features:**
- Session-dependent query detection
- Pronoun-to-entity reference resolution
- Smart query merging with conversation history
- Automatic entity extraction and tracking
- Question-first vector retrieval using dual-embedding architecture
- Graduated context windows for optimal performance

**Use Cases:**
- Follow-up questions without repeating context
- Natural conversational flow
- Pronoun resolution ("it", "that", "them")
- Context-aware semantic search

**Example:**
```
User: "What is WorldTracer?"
Bot:  [Explains WorldTracer is a baggage tracking system...]

User: "How do I configure it?"
→ CCE automatically resolves "it" to "WorldTracer"
→ Generates response about WorldTracer configuration
```

**Documentation:** [conversational-context-engine.md](conversational-context-engine.md)

---

## Quick Links

- **Project Overview:** [../CLAUDE.md](../CLAUDE.md)
- **General Documentation:** [../README.md](../README.md)
- **Source Code:** [../src/](../src/)

## Contributing to Documentation

When adding new feature documentation:

1. Create a new `.md` file in this directory with a descriptive name (e.g., `feature-name.md`)
2. Follow the existing documentation structure:
   - Overview with official feature name
   - Core components
   - Architecture details
   - Code references with file paths and line numbers
   - Examples and use cases
   - Performance characteristics
3. Add an entry to this README.md
4. Reference the documentation in [../CLAUDE.md](../CLAUDE.md) if relevant for Claude Code assistance

## Documentation Standards

- Use clear, concise language
- Include code examples and file references
- Provide architectural diagrams or flows where appropriate
- Document performance characteristics
- Include links to related files using relative paths
- Keep documentation up-to-date with code changes
