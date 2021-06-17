import boto3

subscribers = ['+14088262999']
# Create an SNS client
client = boto3.client(
    "sns",
    aws_access_key_id="AKIAZZWC7PQ7V5RZXDMB",
    aws_secret_access_key="FtGIR5yuui6CDH/bVP/ZFS6p7mvLgcoHmInEU07s",
    region_name="us-east-1"
)

# # Send your sms message.
# client.publish(
#     PhoneNumber="+14088262999",
#     Message="Hello World!"
# )

# Create the topic if it doesn't exist (this is idempotent)
topic = client.create_topic(Name="notifications")
topic_arn = topic['TopicArn']  # get its Amazon Resource Name

# Add SMS Subscribers
for number in subscribers:
    client.subscribe(
        TopicArn=topic_arn,
        Protocol='sms',
        Endpoint=number  # <-- number who'll receive an SMS message.
    )

# Publish a message.
client.publish(Message="Good news everyone!", TopicArn=topic_arn)
