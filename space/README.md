---
title: IEEE Paper Assistant
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: Ask questions about IEEE papers, with cited answers
pinned: false
---

# IEEE Paper Assistant

Upload an IEEE research paper (PDF) and ask questions about its method, results or limitations.
The assistant finds the passages that answer the question (semantic + keyword search, then a
cross-encoder reranker) and Claude replies from those passages only, citing them as [Source N].
Summarize and Review buttons send the whole paper for a structured summary or reviewer-style feedback.

Source code, evaluation and how to run it on your own computer:
https://github.com/TianxinXie123/RAG-Based-IEEE-Paper-Assistant

About this demo:

- Each visitor gets a paper of their own. Uploaded files stay in this Space's temporary storage
  and are deleted when it restarts. The passages used for an answer are sent to the Anthropic API.
- To keep the API costs bounded there are limits per visitor per hour and for all visitors per day.
- After a while without visitors the Space goes to sleep. The first visit after that waits a minute
  or two while it starts.
