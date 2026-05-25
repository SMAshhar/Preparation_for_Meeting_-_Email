import os
from textwrap import dedent
from crewai import Agent
from custom_llm1 import ClaudeLLM, ClaudeThenLocalLLM, CustomLLM1

from tools.ExaSearchTool import ExaSearchTool


# Get local LLM configuration
custom_llm_endpoint = os.getenv("CUSTOM_LLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
custom_llm_model = os.getenv("CUSTOM_LLM_MODEL", "qwen3:8b")
custom_llm_api_key = os.getenv("CUSTOM_LLM_SERVER_API_KEY", "")

# Get Claude API configuration
claude_api_endpoint = os.getenv("CLAUDE_API_ENDPOINT", "https://api.anthropic.com/v1/chat/completions")
claude_model = os.getenv("CLAUDE_MODEL", "claude-3.5")
claude_api_key = os.getenv("CLAUDE_API_KEY", "")

local_llm = CustomLLM1(
    model=custom_llm_model,
    api_key=custom_llm_api_key,
    endpoint=custom_llm_endpoint,
    temperature=0.45,
)

claude_llm = ClaudeLLM(
    model=claude_model,
    api_key=claude_api_key,
    endpoint=claude_api_endpoint,
    temperature=0.45,
)

llm = ClaudeThenLocalLLM(claude_llm=claude_llm, local_llm=local_llm)

class MeetingPreparationAgents():
	def research_agent(self):
		return Agent(
			role='Research Specialist',
			goal='Conduct thorough research on people and companies involved in the meeting',
			tools=ExaSearchTool.tools(),
			backstory=dedent("""\
					As a Research Specialist, your mission is to uncover detailed information
					about the individuals and entities participating in the meeting. Your insights
					will lay the groundwork for strategic meeting preparation."""),
			verbose=True,
			llm=llm,
		)

	def industry_analysis_agent(self):
		return Agent(
			role='Industry Analyst',
			goal='Analyze the current industry trends, challenges, and opportunities',
			tools=ExaSearchTool.tools(),
			backstory=dedent("""\
					As an Industry Analyst, your analysis will identify key trends,
					challenges facing the industry, and potential opportunities that
					could be leveraged during the meeting for strategic advantage."""),
			verbose=True,
			llm=llm,
		)

	def meeting_strategy_agent(self):
		return Agent(
			role='Meeting Strategy Advisor',
			goal='Develop talking points, questions, and strategic angles for the meeting',
			tools=ExaSearchTool.tools(),
			backstory=dedent("""\
					As a Strategy Advisor, your expertise will guide the development of
					talking points, insightful questions, and strategic angles
					to ensure the meeting's objectives are achieved."""),
			verbose=True,
			llm=llm,
		)

	def summary_and_briefing_agent(self):
		return Agent(
			role='Briefing Coordinator',
			goal='Compile all gathered information into a concise, informative briefing document',
			tools=ExaSearchTool.tools(),
			backstory=dedent("""\
					As the Briefing Coordinator, your role is to consolidate the research,
					analysis, and strategic insights."""),
			verbose=True,
			llm=llm,
		)

	def marketing_email_writer_agent(self):
		return Agent(
			role='Email Writer',
			goal='Draft professional and effective emails for the marketing our product to the participants',
			tools=ExaSearchTool.tools(),
			backstory=dedent("""\
					As the Marketing Email Writer, your role is to create effective emails
					that get participants's attention inform them about {the_company} and willing to attend the meeting."""),
			verbose=True,
			llm=llm,
		)
