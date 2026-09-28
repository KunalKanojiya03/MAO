import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
import os
from langchain_groq import ChatGroq

st.set_page_config(page_title="My AI Council", layout="wide")
st.title("🤖 Multi-Agent Council Dashboard")

# Securely load API keys from Streamlit Cloud Secrets
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Initialize the specific models
manager_llm = LLM(model="gemini/gemini-3.1-flash-lite")
worker_llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="llama-3.1-8b-instant")

project_plan = st.text_area("Enter your project brief:", height=150)

if st.button("Deploy the Council"):
    with st.spinner("The council is drafting, coding, and reviewing..."):

        # Define the Agents
        manager = Agent(
            role="Project Manager",
            goal="Analyze the user's project plan and outline a strict technical architecture.",
            backstory="You are an expert software architect managing a team of developers.",
            llm=manager_llm,
            allow_delegation=False,
        )

        developer = Agent(
            role="Senior Developer",
            goal="Write clean Python code based on the Manager's architecture.",
            backstory="You are a fast, precise programmer.",
            llm=worker_llm,
            allow_delegation=False,
        )

        tester = Agent(
            role="QA Tester",
            goal="Review the Developer's code, fix any bugs, and output the final code block.",
            backstory="You are a ruthless code reviewer who catches logic errors.",
            llm=worker_llm,
            allow_delegation=False,
        )

        # Define the Hand-Off Tasks
        task1 = Task(
            description=f"Analyze this plan: {project_plan}. Create a technical blueprint.",
            expected_output="A step-by-step technical blueprint.",
            agent=manager,
        )

        task2 = Task(
            description="Write the code based on the blueprint from the Manager.",
            expected_output="Raw Python code implementing the blueprint.",
            agent=developer,
        )

        task3 = Task(
            description="Review the generated code, fix any bugs, and return the final clean code.",
            expected_output="The final, bug-free Python code block.",
            agent=tester,
        )

        # Assemble and Run
        council = Crew(
            agents=[manager, developer, tester],
            tasks=[task1, task2, task3],
            process=Process.sequential,
        )

        result = council.kickoff()

        st.success("Council Execution Complete!")
        st.markdown("### Final Output")
        st.markdown(result.raw)
