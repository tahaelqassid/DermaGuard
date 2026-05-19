import os
from crewai import Agent, LLM
from tools.cnn_tool import CNNClassifierTool

def build_classifier_agent() -> Agent:
    llm = LLM(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return Agent(
        role="Dermatology CNN Classifier",
        goal="Analyze skin lesion images and return accurate classification results.",
        backstory="You are powered by a ResNet-50 CNN trained on the ISIC dataset.",
        tools=[CNNClassifierTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )