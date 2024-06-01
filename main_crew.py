import os
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from langchain_community.tools import ShellTool


# MODEL = "codellama" # OK
MODEL = "mistral"
# MODEL = "dolphin-llama3"
# MODEL = "gemma"
# MODEL = "llama3"
# MODEL = "llama2"

shell_tool = ShellTool()
# print("1 - " + shell_tool.description)
shell_tool.description = shell_tool.description + f" Use this tool to create and edit files and to compile programs. Arguments: {shell_tool.args}".replace("{", "{{").replace("}", "}}")
# shell_tool.description = shell_tool.description + f" Use this tool to create and edit files and to compile programs. Provide the following arguments: {shell_tool.args}".replace("{", "{{").replace("}", "}}")
# print("2 - " + shell_tool.description)

llm = Ollama(model=MODEL, temperature=0)

coder = Agent(
    llm=llm,
    role='Senior Software Engineer',
    goal='Create software program according to the specified requirements.',
    backstory="""You are a Senior Software Engineer at a leading tech company.
        You have 10 years of experience in working with the C++ programming language.
        Ensure to produce a perfect code.""",
    max_iter = 1,
    verbose=True,
    allow_delegation=False
)

task1 = Task(
    description="""Write a fully working (and without any compilation error) program that can count from 1 to 10. Use C++ programming language.""",
    expected_output="""Provide only the C++ code. Do not add any additional Notes or Explanations.""",
    agent=coder
)

compiler = Agent(
    llm=llm,
    role='C++ Compiler Specialist',
    goal='Ensure all the C++ projects are compiled and built correctly, using various compilers.',
    backstory="""You are a Senior Software Engineer at a leading tech company.
        You have 10 years of experience in working with the C++ programming language and 5 years of experience working with Ubuntu's shell.
        You have a strong understanding of C++ syntax, semantics, and best practices.""",
    tools = [shell_tool],
    max_iter = 5,
    verbose=True,
    allow_delegation=False
)

task2 = Task(
    # description="""print the provided code.""",
    description="""Use the available bash tool to create a file and save the provided code in it.""",
    expected_output="""File containing the provided code, saved in the local Ubuntu machine. Print new file name.""",
    # expected_output="""Runnable file saved on the local Ubuntu machine.""",
    agent=compiler
)

# Do not run the file. 

task3 = Task(
    description="""Use the available bash tool to find any C++ files in the local folder, compile (and build) them.""",
    # expected_output="""New runnable file with containing the provided code. Saved in the local Ubuntu machine. Do not run the file.""",
    expected_output="""Runnable file saved on the local Ubuntu machine.""",
    agent=compiler
)

# Print the provided file name.

crew = Crew(
    agents=[coder, compiler],
    tasks=[task1, task2, task3],
    verbose=2
)

result = crew.kickoff()

print("######################")
print(result)