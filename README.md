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
