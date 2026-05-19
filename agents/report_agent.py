import os
from crewai import Agent, LLM
from tools.report_tool import ReportGeneratorTool

def build_report_agent() -> Agent:
    llm = LLM(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return Agent(
        role="Clinical Report Generator",
        goal="Transform CNN results into structured clinical triage reports.",
        backstory="You are a medical writing AI with expertise in dermatology.",
        tools=[ReportGeneratorTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )