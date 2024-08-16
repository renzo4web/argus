from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import VisionTool
from PIL import Image

vision_tool = VisionTool()

@CrewBase
class ArgusCrew():
    """Argus crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def image_loader(self) -> Agent:
        return Agent(
            config=self.agents_config['image_loader'],
            allow_delegation=False,
            verbose=True
        )

    @agent
    def llm_processor(self) -> Agent:
        return Agent(
            config=self.agents_config['llm_processor'],
            allow_delegation=False,
            tools=[vision_tool],
            verbose=True
        )

    @agent
    def json_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['json_summarizer'],
            allow_delegation=False,
            verbose=True
        )

    @task
    def load_image_task(self) -> Task:
        return Task(
            config=self.tasks_config['load_image_task'],
						action=lambda: Image.open(self.inputs['image_path_url'])  # Open the image file using PIL
        )

    @task
    def process_image_task(self) -> Task:
        return Task(
            config=self.tasks_config['process_image_task'],
						 action=lambda image: self.agents['llm_processor'].tools[0].query_image(
                image=image,
                query="Extract the business name, line items with prices, tax, and total from this receipt image."
            )
        )

    @task
    def summarize_json_task(self) -> Task:
        return Task(
            config=self.tasks_config['summarize_json_task'],
            output_file='receipt_data.json',
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