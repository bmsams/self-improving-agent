"""Lambda function to trigger and wait for CodeBuild during CDK deployment.

Based on the official AWS pattern from amazon-bedrock-agentcore-samples.
"""

import boto3
import json
import logging
import time
import urllib3

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class cfnresponse:
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

    @staticmethod
    def send(event, context, response_status, response_data,
             physical_resource_id=None, reason=None):
        response_url = event["ResponseURL"]

        response_body = {
            "Status": response_status,
            "Reason": reason or f"See CloudWatch Log Stream: {context.log_stream_name}",
            "PhysicalResourceId": physical_resource_id or context.log_stream_name,
            "StackId": event["StackId"],
            "RequestId": event["RequestId"],
            "LogicalResourceId": event["LogicalResourceId"],
            "Data": response_data,
        }

        json_body = json.dumps(response_body)
        headers = {"content-type": "", "content-length": str(len(json_body))}

        try:
            http = urllib3.PoolManager()
            response = http.request("PUT", response_url, headers=headers, body=json_body)
            logger.info(f"CloudFormation response status: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send CloudFormation response: {e}")


def handler(event, context):
    logger.info("Received event: %s", json.dumps(event))

    try:
        if event["RequestType"] == "Delete":
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
            return

        project_name = event["ResourceProperties"]["ProjectName"]
        codebuild_client = boto3.client("codebuild")

        response = codebuild_client.start_build(projectName=project_name)
        build_id = response["build"]["id"]
        logger.info(f"Started build: {build_id}")

        max_wait_time = context.get_remaining_time_in_millis() / 1000 - 30
        start_time = time.time()

        while True:
            if time.time() - start_time > max_wait_time:
                cfnresponse.send(
                    event, context, cfnresponse.FAILED,
                    {"Error": "Build timeout"}
                )
                return

            build_response = codebuild_client.batch_get_builds(ids=[build_id])
            build_status = build_response["builds"][0]["buildStatus"]

            if build_status == "SUCCEEDED":
                logger.info(f"Build {build_id} succeeded")
                cfnresponse.send(
                    event, context, cfnresponse.SUCCESS,
                    {"BuildId": build_id}
                )
                return
            elif build_status in ["FAILED", "FAULT", "STOPPED", "TIMED_OUT"]:
                logger.error(f"Build {build_id} failed: {build_status}")
                cfnresponse.send(
                    event, context, cfnresponse.FAILED,
                    {"Error": f"Build failed: {build_status}"}
                )
                return

            logger.info(f"Build {build_id} status: {build_status}")
            time.sleep(30)

    except Exception as e:
        logger.error("Error: %s", str(e))
        cfnresponse.send(event, context, cfnresponse.FAILED, {"Error": str(e)})
