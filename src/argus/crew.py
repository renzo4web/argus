from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from argus.tools.json_tool import JsonTool
from argus.tools.vision_tool import VisionTool

vision_tool = VisionTool()
json_tool = JsonTool()

@CrewBase
class ArgusCrew():
    """Argus crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def image_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['image_analyst'],
            allow_delegation=False,
            tools=[vision_tool],
            verbose=True
        )

    @agent
    def product_describer(self) -> Agent:
        return Agent(
            config=self.agents_config['product_describer'],
            allow_delegation=False,
            verbose=True
        )

    @agent
    def json_provider(self) -> Agent:
        return Agent(
            config=self.agents_config['json_provider'],
            allow_delegation=False,
            tools=[JsonTool(result_as_answer=True)],
            verbose=True
        )

    @task
    def analyze_image_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_image_task'],
        )

    @task
    def product_description_task(self) -> Task:
        return Task(
            config=self.tasks_config['product_description_task'],
        )
		
    @task
    def structured_json_task(self) -> Task:
        return Task(
            config=self.tasks_config['structured_json_task'],
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