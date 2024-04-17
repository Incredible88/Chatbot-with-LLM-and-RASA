# Get start
```bash
rasa run actions
rasa run
```

## Sent the first message
```bash
$headers = @{
    'Content-type' = 'application/json'
}

$body = @{
    text = 'Hello, World!'
} | ConvertTo-Json

Invoke-WebRequest -Uri 'https://hooks.slack.com/services/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' -Method Post -Headers $headers -Body $body
```
## Screenshot of the APP
![image](https://github.com/Incredible88/Chatbot-with-LLM-and-RASA/assets/60803217/dafbdd6b-83ce-45dc-b7c0-a19c022c48ba)
