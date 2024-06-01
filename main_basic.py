from langchain.agents import create_json_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import Tool
from langchain_community.tools import ShellTool


MODEL = "mistral"
llm = ChatOllama(model=MODEL, format="json", temperature=0.4, num_ctx=4096, mirostat_tau=0.1)

tool_names = ["Terminal"]
shell_tool = Tool(
    name="Terminal",
    func=lambda prompt: {
        "value": ShellTool(
            verbose=True,
            # k=10,
        ).run(prompt)
    },
    description="""Useful for run shell (bash) commands on this Linux machine.
        Arguments: {{'commands': {'title': 'Commands', 'description': 'List of shell commands to run. Deserialized using json.loads', 'anyOf': [...]}}}""".replace("{", "{{").replace("}", "}}")
)
tools = [shell_tool]

SYSTEM = """
<s>[INST] Assistant is a Senior Software Engineer at a leading tech company.
        You have 10 years of experience in working with the C++ programming language and 5 years of experience working with Linux's shell.
        You have a strong understanding of C++ syntax, semantics, and best practices."""

HUMAN = """
TOOLS
------
Assistant can use tools to look up information, which may be helpful in answering the user's original question. These are the tools descriptions:

{tools}

RESPONSE FORMAT INSTRUCTIONS
----------------------------
When responding to user, please output a response in one of two formats:

*Format 1*
Use this format if you need to user a tool. The Markdown code snippet should follow this schema:

```json
{{
    'action': string, (Name of a tool. Must be one of {tool_names})
    'action_input': string (A parameter to send to the tool)
}}
```

*Format 2*
Use this format if you can answer the question. The Markdown code snippet should follow this schema:

```json
{{
    'action': 'Final Answer',
    'action_input': string (The solution)
}}
```

USER'S INPUT
--------------------
Here is the user's input. Remember to respond with the markdown code snippet of JSON blob with a single action, and NOTHING else: [/INST]

{input}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", HUMAN),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_json_chat_agent(
    tools = tools,
    llm = llm,
    prompt = prompt,
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=10, return_messages=True, output_key="output"
)

agent_executor = AgentExecutor(
    agent=agent,
    early_stopping_method="force",
    max_execution_time=180,
    max_iterations=5,
    memory=memory,
    tools=tools,
    handle_parsing_errors=True,
    return_intermediate_steps=False,
    verbose=True,
)

agent_executor.invoke({"input": """Create a program in the C++ programming language that prints numbers from 1 to 10 while using 'std::endl' with 'std::cout'!!!
                        Apply a provided tools to save this program on this Linux machine, compile and run it."""})