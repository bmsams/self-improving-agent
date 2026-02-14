#!/usr/bin/env python3
"""CDK app for Self-Improving Agent on AWS Bedrock AgentCore."""
import aws_cdk as cdk
from agent_stack import SelfImprovingAgentStack

app = cdk.App()
SelfImprovingAgentStack(app, "SelfImprovingAgent")

app.synth()
