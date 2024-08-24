import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, llm
from langchain_openai import ChatOpenAI
from argus.tools.json_tool import JsonTool
from argus.tools.vision_tool import VisionTool
from dotenv import load_dotenv

load_dotenv()

vision_tool = VisionTool()
json_tool = JsonTool()


@CrewBase
class ArgusCrew:
    """Argus crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # LLM

    @llm
    def deepseek_llm(self) -> ChatOpenAI:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY no está configurada en el archivo .env")
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )

    @llm
    def groq_llm(self) -> ChatOpenAI:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY no está configurada en el archivo .env")
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model="gemma-7b-it",
        )

    # AGENTS

    @agent
    def image_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["image_analyst"],
            allow_delegation=False,
            tools=[VisionTool(result_as_answer=True)],
            verbose=True,
        )

    @agent
    def sales_copywriter(self) -> Agent:
        return Agent(
            config=self.agents_config["sales_copywriter"],
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def json_provider(self) -> Agent:
        return Agent(
            config=self.agents_config["json_provider"],
            allow_delegation=False,
            tools=[JsonTool(result_as_answer=True)],
            verbose=True,
        )

    # TASKS

    @task
    def analyze_image_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_image_task"],
        )

    @task
    def product_description_task(self) -> Task:
        return Task(
            config=self.tasks_config["product_description_task"],
        )

    @task
    def structured_json_task(self) -> Task:
        return Task(
            config=self.tasks_config["structured_json_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Argus crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
