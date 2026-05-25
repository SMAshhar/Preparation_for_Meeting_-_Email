import os
import ast
import json
from exa_py import Exa
from crewai.tools import tool

class ExaSearchTool:
	@staticmethod
	def _search_impl(query: str):
		"""Internal implementation of search."""
		return ExaSearchTool._exa().search(f"{query}", use_autoprompt=True, num_results=3)

	@tool
	def search(query: str):
		"""Search for a webpage based on the query."""
		return ExaSearchTool._search_impl(query)

	@staticmethod
	def _find_similar_impl(url: str):
		"""Internal implementation of find_similar."""
		return ExaSearchTool._exa().find_similar(url, num_results=3)

	@staticmethod
	@tool
	def find_similar(url: str):
		"""Search for webpages similar to a given URL.
		The url passed in should be a URL returned from `search`.
		"""
		return ExaSearchTool._find_similar_impl(url)

	@staticmethod
	def _get_contents_impl(ids: str):
		"""Internal implementation of get_contents."""
		try:
			# Try to parse as JSON first (safer)
			try:
				ids = json.loads(ids)
			except (json.JSONDecodeError, TypeError):
				# If JSON parsing fails, try ast.literal_eval as a fallback
				try:
					ids = ast.literal_eval(ids)
				except (ValueError, SyntaxError):
					# If both fail, assume it's a single ID string
					ids = [ids.strip()] if isinstance(ids, str) else []
			
			# Validate that ids is a list
			if not isinstance(ids, list):
				return "Error: IDs must be provided as a list"
			
			# Validate each ID is a string
			if not all(isinstance(id_val, str) for id_val in ids):
				return "Error: All IDs must be strings"
			
			contents = str(ExaSearchTool._exa().get_contents(ids))
			print(contents)
			contents = contents.split("URL:")
			contents = [content[:1000] for content in contents]
			return "\n\n".join(contents)
			
		except Exception as e:
			return f"Error processing IDs: {str(e)}"

	@staticmethod
	@tool
	def get_contents(ids: str):
		"""Get the contents of a webpage.
		The ids must be passed in as a list, a list of ids returned from `search`.
		"""
		return ExaSearchTool._get_contents_impl(ids)

	@staticmethod
	def tools():
		return [ExaSearchTool.search, ExaSearchTool.find_similar, ExaSearchTool.get_contents]

	@staticmethod
	def _exa():
		return Exa(api_key=os.environ["EXA_API_KEY"])
	

if __name__ == "__main__":
	# Test the search tool with a basic query
	result = ExaSearchTool._search_impl("artificial intelligence")
	print("Search Results:")
	print(result)
	


