# """
# LangChain Gemini API Integration for Process Drift Analysis
# This module provides automated analysis of process drift reports using Google's Gemini API
# """

# import os
# import json
# from datetime import datetime
# from typing import Dict, List, Optional, Any
# from dataclasses import dataclass
# from enum import Enum

# # LangChain imports
# from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain.prompts import PromptTemplate
# from langchain.schema import HumanMessage, SystemMessage
# from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
# from langchain.chains import LLMChain, SequentialChain
# from langchain.memory import ConversationBufferMemory

# # Pydantic for structured outputs
# from pydantic import BaseModel, Field
# from typing import List, Dict, Optional

# # =====================================================================================
# # CONFIGURATION AND MODELS
# # =====================================================================================

# class AnalysisType(Enum):
#     STRATEGIC_BUSINESS = "strategic_business"
#     ROOT_CAUSE = "root_cause"
#     PREDICTIVE_INSIGHTS = "predictive_insights"
#     COMPARATIVE = "comparative"
#     IMPLEMENTATION_ROADMAP = "implementation_roadmap"

# @dataclass
# class GeminiConfig:
#     """Configuration for Gemini API"""
#     api_key: str = ""  # Leave empty for user input
#     model_name: str = "gemini-2.5-flash"
#     temperature: float = 0.3
#     max_tokens: int = 2048

# # Pydantic Models for Structured Outputs
# class BusinessRisk(BaseModel):
#     risk_name: str = Field(description="Name of the business risk")
#     severity: str = Field(description="Risk severity: Low, Medium, High, Critical")
#     impact_area: str = Field(description="Business area affected")
#     description: str = Field(description="Detailed description of the risk")
#     mitigation_strategy: str = Field(description="Suggested mitigation approach")

# class StrategicAnalysis(BaseModel):
#     top_risks: List[BusinessRisk] = Field(description="Top 3 business risks identified")
#     operational_improvements: List[str] = Field(description="Priority operational improvements")
#     customer_impact: str = Field(description="Analysis of customer satisfaction impact")
#     executive_summary: str = Field(description="Executive summary for leadership")

# class RootCauseAnalysis(BaseModel):
#     primary_causes: List[str] = Field(description="Primary root causes identified")
#     correlation_insights: List[str] = Field(description="Correlation patterns discovered")
#     external_factors: List[str] = Field(description="External factors contributing to drift")
#     confidence_level: str = Field(description="Confidence in the analysis: Low, Medium, High")

# class PredictiveInsights(BaseModel):
#     future_problems: List[str] = Field(description="Potential problems in next 6 months")
#     resource_adjustments: List[str] = Field(description="Recommended resource allocation changes")
#     early_warnings: List[str] = Field(description="Key indicators to monitor")
#     timeline: str = Field(description="Expected timeline for predicted changes")

# class ImplementationRoadmap(BaseModel):
#     thirty_day_actions: List[str] = Field(description="Actions for first 30 days")
#     sixty_day_actions: List[str] = Field(description="Actions for 30-60 day period")
#     ninety_day_actions: List[str] = Field(description="Actions for 60-90 day period")
#     success_metrics: List[str] = Field(description="KPIs to track progress")
#     resource_requirements: List[str] = Field(description="Required resources and budget")

# # =====================================================================================
# # MAIN ANALYZER CLASS
# # =====================================================================================

# class GeminiDriftAnalyzer:
#     """
#     Main class for analyzing process drift reports using Gemini API via LangChain
#     """
    
#     def __init__(self, config: GeminiConfig):
#         """Initialize the analyzer with Gemini configuration"""
#         self.config = config
#         self.llm = None
#         self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         self._initialize_llm()
#         self._setup_prompts()
        
#     def _initialize_llm(self):
#         """Initialize the Gemini LLM through LangChain"""
#         if not self.config.api_key:
#             raise ValueError("API key is required. Please set the API key in GeminiConfig.")
        
#         # Set environment variable for Google API
#         os.environ["GOOGLE_API_KEY"] = self.config.api_key
        
#         # Initialize Gemini LLM
#         self.llm = ChatGoogleGenerativeAI(
#             model=self.config.model_name,
#             temperature=self.config.temperature,
#             max_tokens=self.config.max_tokens
#         )
        
#         print(f"✅ Successfully initialized Gemini {self.config.model_name}")
    
#     def _setup_prompts(self):
#         """Setup prompt templates for different analysis types"""
        
#         # Strategic Business Analysis Prompt
#         self.strategic_prompt = PromptTemplate(
#             input_variables=["drift_report"],
#             template="""
# You are a senior business analyst specializing in process optimization and risk management.

# TASK: Analyze the following process drift report and provide strategic business insights.

# PROCESS DRIFT REPORT:
# {drift_report}

# ANALYSIS REQUIREMENTS:
# 1. Identify the TOP 3 BUSINESS RISKS with severity levels
# 2. List priority operational improvements (5-7 items)
# 3. Assess impact on customer satisfaction
# 4. Provide executive summary suitable for C-level presentation

# RESPONSE FORMAT:
# Provide a comprehensive analysis that addresses:
# - Risk assessment with severity and mitigation strategies
# - Operational improvement priorities ranked by impact
# - Customer satisfaction implications
# - Strategic recommendations for leadership

# Focus on business value, ROI, and competitive implications.
# """
#         )
        
#         # Root Cause Analysis Prompt
#         self.root_cause_prompt = PromptTemplate(
#             input_variables=["drift_report"],
#             template="""
# You are an expert in business process analysis and organizational change management.

# TASK: Conduct root cause analysis of the process drift patterns in this report.

# PROCESS DRIFT REPORT:
# {drift_report}

# ANALYSIS REQUIREMENTS:
# 1. Identify PRIMARY ROOT CAUSES of the observed drift patterns
# 2. Analyze correlations between different types of drift (activity, resource, performance)
# 3. Consider external factors that might explain these patterns
# 4. Assess confidence level in your analysis

# FOCUS AREAS:
# - Organizational changes or restructuring
# - Technology implementations or system changes
# - Process changes or policy updates
# - Resource constraints or skill gaps
# - Market pressures or regulatory changes
# - Seasonal or cyclical patterns

# Provide evidence-based reasoning for each identified cause.
# """
#         )
        
#         # Predictive Insights Prompt
#         self.predictive_prompt = PromptTemplate(
#             input_variables=["drift_report"],
#             template="""
# You are a business intelligence analyst specializing in predictive analytics and forecasting.

# TASK: Based on the drift patterns in this report, predict future challenges and opportunities.

# PROCESS DRIFT REPORT:
# {drift_report}

# PREDICTION REQUIREMENTS:
# 1. Forecast potential problems in the NEXT 6 MONTHS
# 2. Recommend resource allocation adjustments
# 3. Identify early warning indicators to monitor
# 4. Estimate timeline for predicted changes

# ANALYTICAL APPROACH:
# - Extrapolate current trends
# - Consider accelerating factors
# - Identify intervention points
# - Assess probability of different scenarios

# Provide actionable predictions with confidence indicators.
# """
#         )
        
#         # Implementation Roadmap Prompt
#         self.implementation_prompt = PromptTemplate(
#             input_variables=["drift_report"],
#             template="""
# You are a project management consultant specializing in process improvement implementation.

# TASK: Create a detailed 90-day implementation roadmap based on this drift analysis.

# PROCESS DRIFT REPORT:
# {drift_report}

# ROADMAP REQUIREMENTS:
# 1. 30-DAY ACTIONS: Immediate, high-impact initiatives
# 2. 60-DAY ACTIONS: Medium-term improvements and system changes
# 3. 90-DAY ACTIONS: Strategic initiatives and long-term improvements
# 4. SUCCESS METRICS: KPIs to track progress and ROI
# 5. RESOURCE REQUIREMENTS: Budget, personnel, and technology needs

# PRIORITIZATION CRITERIA:
# - Business impact potential
# - Implementation complexity
# - Resource requirements
# - Risk mitigation value
# - Quick wins vs. strategic improvements

# Provide detailed, actionable plans with clear deliverables and timelines.
# """
#         )
        
#         # Comparative Analysis Prompt
#         self.comparative_prompt = PromptTemplate(
#             input_variables=["report1", "report2"],
#             template="""
# You are a business process improvement consultant specializing in comparative analysis.

# TASK: Compare these two process drift reports and identify the most significant changes.

# FIRST REPORT:
# {report1}

# SECOND REPORT:
# {report2}

# COMPARISON REQUIREMENTS:
# 1. Identify the most significant changes between reports
# 2. Determine which process shows healthier stability patterns
# 3. Highlight improvements or deteriorations
# 4. Recommend focus areas for process optimization

# ANALYSIS FRAMEWORK:
# - Trend analysis (improving vs. deteriorating)
# - Stability assessment
# - Performance comparison
# - Risk profile changes
# - Operational efficiency indicators

# Provide clear recommendations based on the comparative insights.
# """
#         )
    
#     def analyze_strategic_business_impact(self, drift_report: str) -> Dict[str, Any]:
#         """Analyze strategic business impact using Gemini"""
        
#         print("🔍 Analyzing strategic business impact...")
        
#         # Create output parser
#         parser = PydanticOutputParser(pydantic_object=StrategicAnalysis)
        
#         # Create chain
#         chain = LLMChain(
#             llm=self.llm,
#             prompt=self.strategic_prompt,
#             memory=self.memory
#         )
        
#         # Execute analysis
#         try:
#             response = chain.run(drift_report=drift_report)
            
#             # Parse structured output if possible, otherwise return raw response
#             try:
#                 structured_result = parser.parse(response)
#                 return {
#                     "analysis_type": "Strategic Business Analysis",
#                     "structured_output": structured_result.dict(),
#                     "raw_response": response,
#                     "timestamp": datetime.now().isoformat()
#                 }
#             except:
#                 return {
#                     "analysis_type": "Strategic Business Analysis",
#                     "raw_response": response,
#                     "timestamp": datetime.now().isoformat()
#                 }
                
#         except Exception as e:
#             print(f"❌ Error in strategic analysis: {str(e)}")
#             return {"error": str(e), "analysis_type": "Strategic Business Analysis"}
    
#     def analyze_root_causes(self, drift_report: str) -> Dict[str, Any]:
#         """Conduct root cause analysis using Gemini"""
        
#         print("🔍 Conducting root cause analysis...")
        
#         chain = LLMChain(
#             llm=self.llm,
#             prompt=self.root_cause_prompt,
#             memory=self.memory
#         )
        
#         try:
#             response = chain.run(drift_report=drift_report)
#             return {
#                 "analysis_type": "Root Cause Analysis",
#                 "raw_response": response,
#                 "timestamp": datetime.now().isoformat()
#             }
#         except Exception as e:
#             print(f"❌ Error in root cause analysis: {str(e)}")
#             return {"error": str(e), "analysis_type": "Root Cause Analysis"}
    
#     def generate_predictive_insights(self, drift_report: str) -> Dict[str, Any]:
#         """Generate predictive insights using Gemini"""
        
#         print("🔮 Generating predictive insights...")
        
#         chain = LLMChain(
#             llm=self.llm,
#             prompt=self.predictive_prompt,
#             memory=self.memory
#         )
        
#         try:
#             response = chain.run(drift_report=drift_report)
#             return {
#                 "analysis_type": "Predictive Insights",
#                 "raw_response": response,
#                 "timestamp": datetime.now().isoformat()
#             }
#         except Exception as e:
#             print(f"❌ Error in predictive analysis: {str(e)}")
#             return {"error": str(e), "analysis_type": "Predictive Insights"}
    
#     def create_implementation_roadmap(self, drift_report: str) -> Dict[str, Any]:
#         """Create implementation roadmap using Gemini"""
        
#         print("🗺️ Creating implementation roadmap...")
        
#         chain = LLMChain(
#             llm=self.llm,
#             prompt=self.implementation_prompt,
#             memory=self.memory
#         )
        
#         try:
#             response = chain.run(drift_report=drift_report)
#             return {
#                 "analysis_type": "Implementation Roadmap",
#                 "raw_response": response,
#                 "timestamp": datetime.now().isoformat()
#             }
#         except Exception as e:
#             print(f"❌ Error in roadmap creation: {str(e)}")
#             return {"error": str(e), "analysis_type": "Implementation Roadmap"}
    
#     def compare_reports(self, report1: str, report2: str) -> Dict[str, Any]:
#         """Compare two drift reports using Gemini"""
        
#         print("📊 Conducting comparative analysis...")
        
#         chain = LLMChain(
#             llm=self.llm,
#             prompt=self.comparative_prompt,
#             memory=self.memory
#         )
        
#         try:
#             response = chain.run(report1=report1, report2=report2)
#             return {
#                 "analysis_type": "Comparative Analysis",
#                 "raw_response": response,
#                 "timestamp": datetime.now().isoformat()
#             }
#         except Exception as e:
#             print(f"❌ Error in comparative analysis: {str(e)}")
#             return {"error": str(e), "analysis_type": "Comparative Analysis"}
    
#     def comprehensive_analysis(self, drift_report: str) -> Dict[str, Any]:
#         """Run all analysis types on the drift report"""
        
#         print("🚀 Starting comprehensive drift analysis with Gemini...")
        
#         results = {
#             "metadata": {
#                 "analysis_timestamp": datetime.now().isoformat(),
#                 "model_used": self.config.model_name,
#                 "report_length": len(drift_report)
#             },
#             "analyses": {}
#         }
        
#         # Run all analysis types
#         analysis_functions = [
#             ("strategic_business", self.analyze_strategic_business_impact),
#             ("root_cause", self.analyze_root_causes),
#             ("predictive_insights", self.generate_predictive_insights),
#             ("implementation_roadmap", self.create_implementation_roadmap)
#         ]
        
#         for analysis_name, analysis_func in analysis_functions:
#             try:
#                 result = analysis_func(drift_report)
#                 results["analyses"][analysis_name] = result
#                 print(f"✅ Completed {analysis_name} analysis")
#             except Exception as e:
#                 print(f"❌ Failed {analysis_name} analysis: {str(e)}")
#                 results["analyses"][analysis_name] = {"error": str(e)}
        
#         return results
    
#     def save_results(self, results: Dict[str, Any], filename: str = None):
#         """Save analysis results to file"""
        
#         if not filename:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"gemini_drift_analysis_{timestamp}.json"
        
#         with open(filename, 'w', encoding='utf-8') as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
        
#         print(f"💾 Results saved to: {filename}")
#         return filename

# # =====================================================================================
# # USAGE EXAMPLES AND MAIN EXECUTION
# # =====================================================================================

# def main():
#     """Main execution function with examples"""
    
#     # Configuration - USER NEEDS TO SET API KEY
#     config = GeminiConfig(
#         api_key="",  # ⚠️ SET YOUR GEMINI API KEY HERE
#         model_name="gemini-2.5-flash",
#         temperature=0.3,
#         max_tokens=2048
#     )
    
#     # Get API key from user if not set
#     if not config.api_key:
#         config.api_key = input("Please enter your Gemini API key: ").strip()
#         if not config.api_key:
#             print("❌ API key is required to proceed.")
#             return
    
#     try:
#         # Initialize analyzer
#         analyzer = GeminiDriftAnalyzer(config)
        
#         # Example drift report (replace with your actual report)
#         sample_drift_report = """
#         COMPREHENSIVE CONCEPT DRIFT ANALYSIS REPORT
#         Generated on: 2024-07-03 14:30:00

#         EXECUTIVE SUMMARY
#         This report analyzes concept drift in the provided business process event log, focusing on temporal changes in activity patterns, process variants, resource utilization, and performance metrics.

#         1. DATASET OVERVIEW
#         - Total Process Instances (Traces): 31,509
#         - Total Events: 561,470
#         - Unique Activities: 26
#         - Unique Resources: 144
#         - Time Span: 365 days
#         - Average Trace Length: 17.8 events

#         2. TEMPORAL DRIFT ANALYSIS
#         ⚠️ SIGNIFICANT DRIFT DETECTED in 4 window transitions:
#         - Windows 3-4: JS Divergence = 0.156
#         - Windows 7-8: JS Divergence = 0.198
#         - Performance degradation: 23% increase in average case duration
        
#         3. PROCESS VARIANT DRIFT ANALYSIS
#         - New Relations: 12
#         - Disappeared Relations: 8
#         - Significant frequency changes detected
        
#         4. PERFORMANCE DRIFT ANALYSIS
#         - Initial Average Duration: 15.2 hours
#         - Final Average Duration: 18.7 hours
#         - Overall Duration Change: +23.0%
#         ⚠️ SIGNIFICANT PERFORMANCE DRIFT detected
#         """
        
#         # Run comprehensive analysis
#         print("Starting comprehensive Gemini analysis...")
#         results = analyzer.comprehensive_analysis(sample_drift_report)
        
#         # Save results
#         filename = analyzer.save_results(results)
        
#         # Display summary
#         print("\n" + "="*80)
#         print("📋 ANALYSIS SUMMARY")
#         print("="*80)
        
#         for analysis_type, result in results["analyses"].items():
#             if "error" not in result:
#                 print(f"\n✅ {analysis_type.upper()} ANALYSIS:")
#                 print("-" * 50)
#                 # Display first 300 characters of response
#                 response = result.get("raw_response", "No response available")
#                 print(response[:300] + "..." if len(response) > 300 else response)
#             else:
#                 print(f"\n❌ {analysis_type.upper()} ANALYSIS FAILED:")
#                 print(f"Error: {result['error']}")
        
#         print(f"\n💾 Full results saved to: {filename}")
        
#         return analyzer, results
        
#     except Exception as e:
#         print(f"❌ Error in main execution: {str(e)}")
#         return None, None

# # Example usage for specific analysis types
# def example_specific_analysis():
#     """Example of running specific analysis types"""
    
#     # Set your API key here
#     API_KEY = ""  # ⚠️ SET YOUR API KEY
    
#     if not API_KEY:
#         API_KEY = input("Enter your Gemini API key: ")
    
#     config = GeminiConfig(api_key=API_KEY)
#     analyzer = GeminiDriftAnalyzer(config)
    
#     # Load your drift report
#     with open('concept_drift_analysis_report.txt', 'r', encoding='utf-8') as f:
#         drift_report = f.read()
    
#     # Run specific analysis
#     strategic_analysis = analyzer.analyze_strategic_business_impact(drift_report)
    
#     # Save specific results
#     with open('strategic_analysis_results.json', 'w') as f:
#         json.dump(strategic_analysis, f, indent=2)
    
#     print("Strategic analysis complete!")
#     return strategic_analysis

# # For comparing two reports
# def example_comparative_analysis():
#     """Example of comparing two drift reports"""
    
#     API_KEY = ""  # ⚠️ SET YOUR API KEY
#     config = GeminiConfig(api_key=API_KEY)
#     analyzer = GeminiDriftAnalyzer(config)
    
#     # Load two reports
#     with open('report1.txt', 'r') as f:
#         report1 = f.read()
    
#     with open('report2.txt', 'r') as f:
#         report2 = f.read()
    
#     # Compare reports
#     comparison = analyzer.compare_reports(report1, report2)
    
#     print("Comparative analysis complete!")
#     return comparison

# if __name__ == "__main__":
#     analyzer, results = main()


"""
LangChain Gemini API Integration for Process Drift Analysis
This module provides automated analysis of process drift reports using Google's Gemini API
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

# Pydantic for structured outputs
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# =====================================================================================
# CONFIGURATION AND MODELS
# =====================================================================================

class AnalysisType(Enum):
    STRATEGIC_BUSINESS = "strategic_business"
    ROOT_CAUSE = "root_cause"
    PREDICTIVE_INSIGHTS = "predictive_insights"
    COMPARATIVE = "comparative"
    IMPLEMENTATION_ROADMAP = "implementation_roadmap"

@dataclass
class GeminiConfig:
    """Configuration for Gemini API"""
    api_key: str = ""  # Leave empty for user input
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.3
    max_tokens: int = 2048

# Pydantic Models for Structured Outputs
class BusinessRisk(BaseModel):
    risk_name: str = Field(description="Name of the business risk")
    severity: str = Field(description="Risk severity: Low, Medium, High, Critical")
    impact_area: str = Field(description="Business area affected")
    description: str = Field(description="Detailed description of the risk")
    mitigation_strategy: str = Field(description="Suggested mitigation approach")

class StrategicAnalysis(BaseModel):
    top_risks: List[BusinessRisk] = Field(description="Top 3 business risks identified")
    operational_improvements: List[str] = Field(description="Priority operational improvements")
    customer_impact: str = Field(description="Analysis of customer satisfaction impact")
    executive_summary: str = Field(description="Executive summary for leadership")

class RootCauseAnalysis(BaseModel):
    primary_causes: List[str] = Field(description="Primary root causes identified")
    correlation_insights: List[str] = Field(description="Correlation patterns discovered")
    external_factors: List[str] = Field(description="External factors contributing to drift")
    confidence_level: str = Field(description="Confidence in the analysis: Low, Medium, High")

class PredictiveInsights(BaseModel):
    future_problems: List[str] = Field(description="Potential problems in next 6 months")
    resource_adjustments: List[str] = Field(description="Recommended resource allocation changes")
    early_warnings: List[str] = Field(description="Key indicators to monitor")
    timeline: str = Field(description="Expected timeline for predicted changes")

class ImplementationRoadmap(BaseModel):
    thirty_day_actions: List[str] = Field(description="Actions for first 30 days")
    sixty_day_actions: List[str] = Field(description="Actions for 30-60 day period")
    ninety_day_actions: List[str] = Field(description="Actions for 60-90 day period")
    success_metrics: List[str] = Field(description="KPIs to track progress")
    resource_requirements: List[str] = Field(description="Required resources and budget")

# =====================================================================================
# MAIN ANALYZER CLASS
# =====================================================================================

class GeminiDriftAnalyzer:
    """
    Main class for analyzing process drift reports using Gemini API via LangChain
    """
    
    def __init__(self, config: GeminiConfig):
        """Initialize the analyzer with Gemini configuration"""
        self.config = config
        self.llm = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self._initialize_llm()
        self._setup_prompts()
        
    def _initialize_llm(self):
        """Initialize the Gemini LLM through LangChain"""
        if not self.config.api_key:
            raise ValueError("API key is required. Please set the API key in GeminiConfig.")
        
        # Set environment variable for Google API
        os.environ["GOOGLE_API_KEY"] = self.config.api_key
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        print(f"✅ Successfully initialized Gemini {self.config.model_name}")
    
    def _setup_prompts(self):
        """Setup prompt templates for different analysis types"""
        
        # Strategic Business Analysis Prompt
        self.strategic_prompt = PromptTemplate(
            input_variables=["drift_report"],
            template="""
You are a senior business analyst specializing in process optimization and risk management.

TASK: Analyze the following process drift report and provide strategic business insights.

PROCESS DRIFT REPORT:
{drift_report}

ANALYSIS REQUIREMENTS:
1. Identify the TOP 3 BUSINESS RISKS with severity levels
2. List priority operational improvements (5-7 items)
3. Assess impact on customer satisfaction
4. Provide executive summary suitable for C-level presentation

RESPONSE FORMAT:
Provide a comprehensive analysis that addresses:
- Risk assessment with severity and mitigation strategies
- Operational improvement priorities ranked by impact
- Customer satisfaction implications
- Strategic recommendations for leadership

Focus on business value, ROI, and competitive implications.
"""
        )
        
        # Root Cause Analysis Prompt
        self.root_cause_prompt = PromptTemplate(
            input_variables=["drift_report"],
            template="""
You are an expert in business process analysis and organizational change management.

TASK: Conduct root cause analysis of the process drift patterns in this report.

PROCESS DRIFT REPORT:
{drift_report}

ANALYSIS REQUIREMENTS:
1. Identify PRIMARY ROOT CAUSES of the observed drift patterns
2. Analyze correlations between different types of drift (activity, resource, performance)
3. Consider external factors that might explain these patterns
4. Assess confidence level in your analysis

FOCUS AREAS:
- Organizational changes or restructuring
- Technology implementations or system changes
- Process changes or policy updates
- Resource constraints or skill gaps
- Market pressures or regulatory changes
- Seasonal or cyclical patterns

Provide evidence-based reasoning for each identified cause.
"""
        )
        
        # Predictive Insights Prompt
        self.predictive_prompt = PromptTemplate(
            input_variables=["drift_report"],
            template="""
You are a business intelligence analyst specializing in predictive analytics and forecasting.

TASK: Based on the drift patterns in this report, predict future challenges and opportunities.

PROCESS DRIFT REPORT:
{drift_report}

PREDICTION REQUIREMENTS:
1. Forecast potential problems in the NEXT 6 MONTHS
2. Recommend resource allocation adjustments
3. Identify early warning indicators to monitor
4. Estimate timeline for predicted changes

ANALYTICAL APPROACH:
- Extrapolate current trends
- Consider accelerating factors
- Identify intervention points
- Assess probability of different scenarios

Provide actionable predictions with confidence indicators.
"""
        )
        
        # Implementation Roadmap Prompt
        self.implementation_prompt = PromptTemplate(
            input_variables=["drift_report"],
            template="""
You are a project management consultant specializing in process improvement implementation.

TASK: Create a detailed 90-day implementation roadmap based on this drift analysis.

PROCESS DRIFT REPORT:
{drift_report}

ROADMAP REQUIREMENTS:
1. 30-DAY ACTIONS: Immediate, high-impact initiatives
2. 60-DAY ACTIONS: Medium-term improvements and system changes
3. 90-DAY ACTIONS: Strategic initiatives and long-term improvements
4. SUCCESS METRICS: KPIs to track progress and ROI
5. RESOURCE REQUIREMENTS: Budget, personnel, and technology needs

PRIORITIZATION CRITERIA:
- Business impact potential
- Implementation complexity
- Resource requirements
- Risk mitigation value
- Quick wins vs. strategic improvements

Provide detailed, actionable plans with clear deliverables and timelines.
"""
        )
        
        # Comparative Analysis Prompt
        self.comparative_prompt = PromptTemplate(
            input_variables=["report1", "report2"],
            template="""
You are a business process improvement consultant specializing in comparative analysis.

TASK: Compare these two process drift reports and identify the most significant changes.

FIRST REPORT:
{report1}

SECOND REPORT:
{report2}

COMPARISON REQUIREMENTS:
1. Identify the most significant changes between reports
2. Determine which process shows healthier stability patterns
3. Highlight improvements or deteriorations
4. Recommend focus areas for process optimization

ANALYSIS FRAMEWORK:
- Trend analysis (improving vs. deteriorating)
- Stability assessment
- Performance comparison
- Risk profile changes
- Operational efficiency indicators

Provide clear recommendations based on the comparative insights.
"""
        )
    
    def analyze_strategic_business_impact(self, drift_report: str) -> Dict[str, Any]:
        """Analyze strategic business impact using Gemini"""
        
        print("🔍 Analyzing strategic business impact...")
        
        # Create output parser
        parser = PydanticOutputParser(pydantic_object=StrategicAnalysis)
        
        # Create chain
        chain = LLMChain(
            llm=self.llm,
            prompt=self.strategic_prompt,
            memory=self.memory
        )
        
        # Execute analysis
        try:
            response = chain.run(drift_report=drift_report)
            
            # Parse structured output if possible, otherwise return raw response
            try:
                structured_result = parser.parse(response)
                return {
                    "analysis_type": "Strategic Business Analysis",
                    "structured_output": structured_result.dict(),
                    "raw_response": response,
                    "timestamp": datetime.now().isoformat()
                }
            except:
                return {
                    "analysis_type": "Strategic Business Analysis",
                    "raw_response": response,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Error in strategic analysis: {str(e)}")
            return {"error": str(e), "analysis_type": "Strategic Business Analysis"}
    
    def analyze_root_causes(self, drift_report: str) -> Dict[str, Any]:
        """Conduct root cause analysis using Gemini"""
        
        print("🔍 Conducting root cause analysis...")
        
        chain = LLMChain(
            llm=self.llm,
            prompt=self.root_cause_prompt,
            memory=self.memory
        )
        
        try:
            response = chain.run(drift_report=drift_report)
            return {
                "analysis_type": "Root Cause Analysis",
                "raw_response": response,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error in root cause analysis: {str(e)}")
            return {"error": str(e), "analysis_type": "Root Cause Analysis"}
    
    def generate_predictive_insights(self, drift_report: str) -> Dict[str, Any]:
        """Generate predictive insights using Gemini"""
        
        print("🔮 Generating predictive insights...")
        
        chain = LLMChain(
            llm=self.llm,
            prompt=self.predictive_prompt,
            memory=self.memory
        )
        
        try:
            response = chain.run(drift_report=drift_report)
            return {
                "analysis_type": "Predictive Insights",
                "raw_response": response,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error in predictive analysis: {str(e)}")
            return {"error": str(e), "analysis_type": "Predictive Insights"}
    
    def create_implementation_roadmap(self, drift_report: str) -> Dict[str, Any]:
        """Create implementation roadmap using Gemini"""
        
        print("🗺️ Creating implementation roadmap...")
        
        chain = LLMChain(
            llm=self.llm,
            prompt=self.implementation_prompt,
            memory=self.memory
        )
        
        try:
            response = chain.run(drift_report=drift_report)
            return {
                "analysis_type": "Implementation Roadmap",
                "raw_response": response,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error in roadmap creation: {str(e)}")
            return {"error": str(e), "analysis_type": "Implementation Roadmap"}
    
    def compare_reports(self, report1: str, report2: str) -> Dict[str, Any]:
        """Compare two drift reports using Gemini"""
        
        print("📊 Conducting comparative analysis...")
        
        chain = LLMChain(
            llm=self.llm,
            prompt=self.comparative_prompt,
            memory=self.memory
        )
        
        try:
            response = chain.run(report1=report1, report2=report2)
            return {
                "analysis_type": "Comparative Analysis",
                "raw_response": response,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error in comparative analysis: {str(e)}")
            return {"error": str(e), "analysis_type": "Comparative Analysis"}
    
    def comprehensive_analysis(self, drift_report: str) -> Dict[str, Any]:
        """Run all analysis types on the drift report"""
        
        print("🚀 Starting comprehensive drift analysis with Gemini...")
        
        results = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": self.config.model_name,
                "report_length": len(drift_report)
            },
            "analyses": {}
        }
        
        # Run all analysis types
        analysis_functions = [
            ("strategic_business", self.analyze_strategic_business_impact),
            ("root_cause", self.analyze_root_causes),
            ("predictive_insights", self.generate_predictive_insights),
            ("implementation_roadmap", self.create_implementation_roadmap)
        ]
        
        for analysis_name, analysis_func in analysis_functions:
            try:
                result = analysis_func(drift_report)
                results["analyses"][analysis_name] = result
                print(f"✅ Completed {analysis_name} analysis")
            except Exception as e:
                print(f"❌ Failed {analysis_name} analysis: {str(e)}")
                results["analyses"][analysis_name] = {"error": str(e)}
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save analysis results to JSON file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gemini_drift_analysis_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filename}")
        return filename
    
    def save_results_markdown(self, results: Dict[str, Any], filename: str = None):
        """Save analysis results to Markdown file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gemini_drift_analysis_{timestamp}.md"
        
        # Generate markdown content
        markdown_content = self._generate_markdown_report(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"📝 Markdown results saved to: {filename}")
        return filename
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate formatted markdown report from results"""
        
        metadata = results.get("metadata", {})
        analyses = results.get("analyses", {})
        
        # Start building markdown
        md_lines = []
        
        # Header
        md_lines.append("# Process Drift Analysis Report")
        md_lines.append("")
        md_lines.append("## Analysis Overview")
        md_lines.append("")
        
        # Metadata section
        md_lines.append("### Report Metadata")
        md_lines.append("")
        md_lines.append(f"- **Analysis Timestamp**: {metadata.get('analysis_timestamp', 'N/A')}")
        md_lines.append(f"- **Model Used**: {metadata.get('model_used', 'N/A')}")
        md_lines.append(f"- **Report Length**: {metadata.get('report_length', 'N/A')} characters")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # Analysis sections
        analysis_titles = {
            "strategic_business": "Strategic Business Impact Analysis",
            "root_cause": "Root Cause Analysis",
            "predictive_insights": "Predictive Insights",
            "implementation_roadmap": "Implementation Roadmap"
        }
        
        for analysis_key, analysis_data in analyses.items():
            title = analysis_titles.get(analysis_key, analysis_key.title().replace("_", " "))
            
            md_lines.append(f"## {title}")
            md_lines.append("")
            
            if "error" in analysis_data:
                md_lines.append(f"❌ **Error**: {analysis_data['error']}")
                md_lines.append("")
            else:
                # Add timestamp if available
                if "timestamp" in analysis_data:
                    md_lines.append(f"*Generated at: {analysis_data['timestamp']}*")
                    md_lines.append("")
                
                # Add structured output if available
                if "structured_output" in analysis_data:
                    md_lines.append("### Structured Analysis")
                    md_lines.append("")
                    structured = analysis_data["structured_output"]
                    
                    if analysis_key == "strategic_business":
                        # Format strategic business analysis
                        md_lines.extend(self._format_strategic_analysis(structured))
                    else:
                        # Generic structured format
                        for key, value in structured.items():
                            md_lines.append(f"**{key.title().replace('_', ' ')}**:")
                            if isinstance(value, list):
                                for item in value:
                                    md_lines.append(f"- {item}")
                            else:
                                md_lines.append(f"{value}")
                            md_lines.append("")
                
                # Add raw response
                if "raw_response" in analysis_data:
                    md_lines.append("### Detailed Analysis")
                    md_lines.append("")
                    raw_response = analysis_data["raw_response"]
                    
                    # Format the raw response with proper markdown
                    formatted_response = self._format_raw_response(raw_response)
                    md_lines.append(formatted_response)
                    md_lines.append("")
            
            md_lines.append("---")
            md_lines.append("")
        
        # Footer
        md_lines.append("## Report Generation Info")
        md_lines.append("")
        md_lines.append("This report was generated using Google Gemini API via LangChain.")
        md_lines.append(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append("")
        
        return "\n".join(md_lines)
    
    def _format_strategic_analysis(self, structured: Dict[str, Any]) -> List[str]:
        """Format strategic business analysis for markdown"""
        lines = []
        
        # Top risks
        if "top_risks" in structured:
            lines.append("#### Top Business Risks")
            lines.append("")
            for i, risk in enumerate(structured["top_risks"], 1):
                lines.append(f"##### Risk {i}: {risk.get('risk_name', 'Unknown Risk')}")
                lines.append(f"- **Severity**: {risk.get('severity', 'N/A')}")
                lines.append(f"- **Impact Area**: {risk.get('impact_area', 'N/A')}")
                lines.append(f"- **Description**: {risk.get('description', 'N/A')}")
                lines.append(f"- **Mitigation Strategy**: {risk.get('mitigation_strategy', 'N/A')}")
                lines.append("")
        
        # Operational improvements
        if "operational_improvements" in structured:
            lines.append("#### Operational Improvements")
            lines.append("")
            for improvement in structured["operational_improvements"]:
                lines.append(f"- {improvement}")
            lines.append("")
        
        # Customer impact
        if "customer_impact" in structured:
            lines.append("#### Customer Impact")
            lines.append("")
            lines.append(structured["customer_impact"])
            lines.append("")
        
        # Executive summary
        if "executive_summary" in structured:
            lines.append("#### Executive Summary")
            lines.append("")
            lines.append(structured["executive_summary"])
            lines.append("")
        
        return lines
    
    def _format_raw_response(self, raw_response: str) -> str:
        """Format raw response for better markdown display"""
        # Clean up the response and add proper formatting
        response = raw_response.strip()
        
        # Add code blocks for any JSON-like content
        if response.startswith('{') and response.endswith('}'):
            return f"```json\n{response}\n```"
        
        # Otherwise, return as-is with proper paragraph breaks
        return response

# =====================================================================================
# USAGE EXAMPLES AND MAIN EXECUTION
# =====================================================================================

def main():
    """Main execution function with examples"""
    
    # Configuration - USER NEEDS TO SET API KEY
    config = GeminiConfig(
        api_key="",  # ⚠️ SET YOUR GEMINI API KEY HERE
        model_name="gemini-2.5-flash",
        temperature=0.3,
        max_tokens=2048
    )
    
    # Get API key from user if not set
    if not config.api_key:
        config.api_key = input("Please enter your Gemini API key: ").strip()
        if not config.api_key:
            print("❌ API key is required to proceed.")
            return
    
    try:
        # Initialize analyzer
        analyzer = GeminiDriftAnalyzer(config)
        
        # Example drift report (replace with your actual report)
        sample_drift_report = """
        COMPREHENSIVE CONCEPT DRIFT ANALYSIS REPORT
        Generated on: 2024-07-03 14:30:00

        EXECUTIVE SUMMARY
        This report analyzes concept drift in the provided business process event log, focusing on temporal changes in activity patterns, process variants, resource utilization, and performance metrics.

        1. DATASET OVERVIEW
        - Total Process Instances (Traces): 31,509
        - Total Events: 561,470
        - Unique Activities: 26
        - Unique Resources: 144
        - Time Span: 365 days
        - Average Trace Length: 17.8 events

        2. TEMPORAL DRIFT ANALYSIS
        ⚠️ SIGNIFICANT DRIFT DETECTED in 4 window transitions:
        - Windows 3-4: JS Divergence = 0.156
        - Windows 7-8: JS Divergence = 0.198
        - Performance degradation: 23% increase in average case duration
        
        3. PROCESS VARIANT DRIFT ANALYSIS
        - New Relations: 12
        - Disappeared Relations: 8
        - Significant frequency changes detected
        
        4. PERFORMANCE DRIFT ANALYSIS
        - Initial Average Duration: 15.2 hours
        - Final Average Duration: 18.7 hours
        - Overall Duration Change: +23.0%
        ⚠️ SIGNIFICANT PERFORMANCE DRIFT detected
        """
        
        # Run comprehensive analysis
        print("Starting comprehensive Gemini analysis...")
        results = analyzer.comprehensive_analysis(sample_drift_report)
        
        # Save results in both formats
        json_filename = analyzer.save_results(results)
        md_filename = analyzer.save_results_markdown(results)
        
        # Display summary
        print("\n" + "="*80)
        print("📋 ANALYSIS SUMMARY")
        print("="*80)
        
        for analysis_type, result in results["analyses"].items():
            if "error" not in result:
                print(f"\n✅ {analysis_type.upper()} ANALYSIS:")
                print("-" * 50)
                # Display first 300 characters of response
                response = result.get("raw_response", "No response available")
                print(response[:300] + "..." if len(response) > 300 else response)
            else:
                print(f"\n❌ {analysis_type.upper()} ANALYSIS FAILED:")
                print(f"Error: {result['error']}")
        
        print(f"\n💾 Results saved to:")
        print(f"   📄 JSON: {json_filename}")
        print(f"   📝 Markdown: {md_filename}")
        
        return analyzer, results
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        return None, None

# Example usage for specific analysis types
def example_specific_analysis():
    """Example of running specific analysis types"""
    
    # Set your API key here
    API_KEY = ""  # ⚠️ SET YOUR API KEY
    
    if not API_KEY:
        API_KEY = input("Enter your Gemini API key: ")
    
    config = GeminiConfig(api_key=API_KEY)
    analyzer = GeminiDriftAnalyzer(config)
    
    # Load your drift report
    with open('concept_drift_analysis_report.txt', 'r', encoding='utf-8') as f:
        drift_report = f.read()
    
    # Run specific analysis
    strategic_analysis = analyzer.analyze_strategic_business_impact(drift_report)
    
    # Save specific results in markdown format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_filename = f"strategic_analysis_{timestamp}.md"
    
    # Create a results structure for markdown generation
    results = {
        "metadata": {
            "analysis_timestamp": datetime.now().isoformat(),
            "model_used": analyzer.config.model_name,
            "report_length": len(drift_report)
        },
        "analyses": {
            "strategic_business": strategic_analysis
        }
    }
    
    analyzer.save_results_markdown(results, md_filename)
    
    print(f"Strategic analysis complete! Saved to: {md_filename}")
    return strategic_analysis

# For comparing two reports
def example_comparative_analysis():
    """Example of comparing two drift reports"""
    
    API_KEY = ""  # ⚠️ SET YOUR API KEY
    config = GeminiConfig(api_key=API_KEY)
    analyzer = GeminiDriftAnalyzer(config)
    
    # Load two reports
    with open('report1.txt', 'r') as f:
        report1 = f.read()
    
    with open('report2.txt', 'r') as f:
        report2 = f.read()
    
    # Compare reports
    comparison = analyzer.compare_reports(report1, report2)
    
    # Save comparison in markdown format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_filename = f"comparative_analysis_{timestamp}.md"
    
    results = {
        "metadata": {
            "analysis_timestamp": datetime.now().isoformat(),
            "model_used": analyzer.config.model_name,
            "report_length": len(report1) + len(report2)
        },
        "analyses": {
            "comparative": comparison
        }
    }
    
    analyzer.save_results_markdown(results, md_filename)
    
    print(f"Comparative analysis complete! Saved to: {md_filename}")
    return comparison

if __name__ == "__main__":
    analyzer, results = main()