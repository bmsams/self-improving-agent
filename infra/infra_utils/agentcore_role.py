"""IAM Role for AgentCore Runtime execution.

Based on the official AWS pattern from amazon-bedrock-agentcore-samples.
Service principal: bedrock-agentcore.amazonaws.com
"""

from aws_cdk import aws_iam as iam, Stack
from constructs import Construct


class AgentCoreRole(iam.Role):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        region = Stack.of(scope).region
        account_id = Stack.of(scope).account

        statements = [
            # ECR image access for container pull
            iam.PolicyStatement(
                sid="ECRImageAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchCheckLayerAvailability",
                ],
                resources=[f"arn:aws:ecr:{region}:{account_id}:repository/*"],
            ),
            iam.PolicyStatement(
                sid="ECRTokenAccess",
                effect=iam.Effect.ALLOW,
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            ),
            # CloudWatch Logs for agent runtime
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:DescribeLogStreams",
                    "logs:CreateLogGroup",
                    "logs:DescribeLogGroups",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=[
                    f"arn:aws:logs:{region}:{account_id}:log-group:/aws/bedrock-agentcore/runtimes/*"
                ],
            ),
            # X-Ray tracing for observability
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules",
                    "xray:GetSamplingTargets",
                ],
                resources=["*"],
            ),
            # CloudWatch metrics
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={
                    "StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}
                },
            ),
            # Workload identity tokens
            iam.PolicyStatement(
                sid="GetAgentAccessToken",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock-agentcore:GetWorkloadAccessToken",
                    "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                    "bedrock-agentcore:GetWorkloadAccessTokenForUserId",
                ],
                resources=[
                    f"arn:aws:bedrock-agentcore:{region}:{account_id}:workload-identity-directory/default",
                    f"arn:aws:bedrock-agentcore:{region}:{account_id}:workload-identity-directory/default/workload-identity/*",
                ],
            ),
            # Bedrock model invocation (for Claude via Bedrock)
            iam.PolicyStatement(
                sid="BedrockModelInvocation",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:{region}:{account_id}:*",
                ],
            ),
            # Code Interpreter permissions
            iam.PolicyStatement(
                sid="CodeInterpreterAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock-agentcore:StartCodeInterpreterSession",
                    "bedrock-agentcore:StopCodeInterpreterSession",
                    "bedrock-agentcore:InvokeCodeInterpreter",
                    "bedrock-agentcore:ListCodeInterpreterSessions",
                ],
                resources=[
                    f"arn:aws:bedrock-agentcore:{region}:{account_id}:code-interpreter/*"
                ],
            ),
            # Memory permissions
            iam.PolicyStatement(
                sid="MemoryAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock-agentcore:ListEvents",
                    "bedrock-agentcore:PutEvents",
                    "bedrock-agentcore:GetEvents",
                    "bedrock-agentcore:CreateEvent",
                ],
                resources=[
                    f"arn:aws:bedrock-agentcore:{region}:{account_id}:memory/*"
                ],
            ),
            # STS for identity
            iam.PolicyStatement(
                sid="STSAccess",
                effect=iam.Effect.ALLOW,
                actions=["sts:GetCallerIdentity"],
                resources=["*"],
            ),
        ]

        super().__init__(
            scope,
            construct_id,
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
            inline_policies={
                "AgentCorePolicy": iam.PolicyDocument(statements=statements)
            },
            **kwargs,
        )
