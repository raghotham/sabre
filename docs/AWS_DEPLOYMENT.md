# AWS Deployment Guide

This guide explains how to deploy SABRE to AWS using GitHub Actions for automated deployment.

## Architecture

```
┌──────────────────────┐
│ GitHub Repository    │
│  ├─ ui/ (Next.js)    │  → AWS Amplify (Frontend)
│  └─ sabre/ (Python)  │  → AWS App Runner (Backend)
└──────────────────────┘
         │
         │ Push to main
         ↓
  ┌─────────────────┐
  │ GitHub Actions  │
  │  ├─ deploy-server.yml   → Builds Docker, deploys to App Runner
  │  └─ deploy-ui.yml       → Triggers Amplify build
  └─────────────────┘
```

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured locally (for initial setup)
3. **GitHub Repository** with this code

## Part 1: Setup GitHub Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

### Required Secrets

| Secret Name | Description | How to Get |
|------------|-------------|------------|
| `AWS_ACCESS_KEY_ID` | AWS access key | AWS IAM Console → Create Access Key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Same as above |
| `OPENAI_API_KEY` | OpenAI API key | platform.openai.com → API Keys |
| `AMPLIFY_APP_ID` | Amplify app ID | Created in Part 3 below |

### Getting AWS Credentials

```bash
# Create IAM user with these policies:
# - AmazonEC2ContainerRegistryFullAccess
# - AWSAppRunnerFullAccess
# - AdministratorAccess-Amplify

aws iam create-user --user-name github-actions-sabre
aws iam attach-user-policy --user-name github-actions-sabre \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess
aws iam attach-user-policy --user-name github-actions-sabre \
  --policy-arn arn:aws:iam::aws:policy/AWSAppRunnerFullAccess
aws iam attach-user-policy --user-name github-actions-sabre \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess-Amplify

# Create access key
aws iam create-access-key --user-name github-actions-sabre
```

## Part 2: Deploy SABRE Server (AWS App Runner)

The server deploys automatically when you push to `main` branch.

### Manual Deployment

Trigger manually via GitHub Actions:
1. Go to Actions tab → "Deploy SABRE Server to AWS App Runner"
2. Click "Run workflow" → Select branch → Run

### What Happens

1. GitHub Actions builds Docker image
2. Pushes to Amazon ECR
3. Creates/updates App Runner service
4. Service becomes available at: `https://[random].us-east-1.awsapprunner.com`

### Monitoring

```bash
# Check service status
aws apprunner list-services --region us-east-1

# Get service URL
aws apprunner describe-service --service-arn [ARN] --region us-east-1 \
  | jq -r '.Service.ServiceUrl'
```

## Part 3: Deploy SABRE UI (AWS Amplify)

Amplify requires one-time setup via AWS CLI:

### Initial Setup

```bash
# 1. Create Amplify app
aws amplify create-app \
  --name sabre-ui \
  --repository https://github.com/YOUR_USERNAME/sabre \
  --access-token [GITHUB_PERSONAL_ACCESS_TOKEN] \
  --region us-east-1

# Save the returned App ID
export AMPLIFY_APP_ID="[app-id-from-output]"

# 2. Create main branch
aws amplify create-branch \
  --app-id $AMPLIFY_APP_ID \
  --branch-name main \
  --enable-auto-build \
  --region us-east-1

# 3. Set build spec path
aws amplify update-app \
  --app-id $AMPLIFY_APP_ID \
  --build-spec "$(cat ui/amplify.yml)" \
  --region us-east-1

# 4. Set environment variable
aws amplify update-branch \
  --app-id $AMPLIFY_APP_ID \
  --branch-name main \
  --environment-variables NEXT_PUBLIC_API_URL=https://[YOUR-APP-RUNNER-URL] \
  --region us-east-1

# 5. Add App ID to GitHub Secrets
# Copy the App ID and add it as AMPLIFY_APP_ID secret in GitHub
```

### Get GitHub Personal Access Token

AWS Amplify needs a GitHub token to:
- Pull code from your repository for builds
- Set up webhooks for automatic deployments
- Update build/deployment status

**Recommended: Fine-Grained Token (better security)**

1. Navigate to GitHub token settings:
   - Go to https://github.com/settings/personal-access-tokens/new
   - Or: GitHub.com → Profile (top right) → Settings → Developer settings → Personal access tokens → **Fine-grained tokens**

2. Click "Generate new token"

3. Configure the token:
   - **Token name**: `AWS Amplify - SABRE`
   - **Expiration**: 90 days (or your preference)
   - **Repository access**: "Only select repositories" → Choose your `sabre` repository
   - **Permissions** (Repository permissions):
     - **Contents**: Read and write
     - **Metadata**: Read-only (automatically included)
     - **Webhooks**: Read and write
     - **Commit statuses**: Read and write

4. Click "Generate token"

5. **Copy the token immediately** - GitHub shows it only once!
   - Token format: `github_pat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - Store it securely - you'll need it for the `aws amplify create-app` command below

**Alternative: Classic Token (if fine-grained doesn't work)**

If you encounter issues with fine-grained tokens:

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Configure:
   - **Note**: `AWS Amplify - SABRE`
   - **Expiration**: 90 days or No expiration
   - **Scopes**: Check `repo` (Full control of private repositories)
4. Generate and copy immediately (format: `ghp_xxxxxxxxxxxx...`)

### After Setup

Deployments trigger automatically on push to `main` when `ui/` changes.

## Part 4: Environment Variables

### Server (App Runner)

Set via GitHub secret or AWS CLI:

```bash
aws apprunner update-service \
  --service-arn [ARN] \
  --source-configuration '{
    "ImageRepository": {
      "ImageConfiguration": {
        "RuntimeEnvironmentVariables": {
          "OPENAI_API_KEY": "sk-...",
          "OPENAI_MODEL": "gpt-4o",
          "PORT": "8011"
        }
      }
    }
  }' \
  --region us-east-1
```

### UI (Amplify)

Set via AWS CLI:

```bash
aws amplify update-branch \
  --app-id $AMPLIFY_APP_ID \
  --branch-name main \
  --environment-variables \
    NEXT_PUBLIC_API_URL=https://[YOUR-APP-RUNNER-URL] \
  --region us-east-1
```

## Part 5: Verification

### Test Server

```bash
# Get server URL
SERVER_URL=$(aws apprunner list-services --region us-east-1 \
  | jq -r '.ServiceSummaryList[] | select(.ServiceName=="sabre-server") | .ServiceUrl')

# Test health endpoint
curl https://$SERVER_URL/health

# Test session list
curl https://$SERVER_URL/v1/sessions
```

### Test UI

```bash
# Get UI URL
UI_URL=$(aws amplify get-app --app-id $AMPLIFY_APP_ID --region us-east-1 \
  | jq -r '.app.defaultDomain')

echo "UI: https://main.$UI_URL"
```

## Part 6: Continuous Deployment

Once set up, deployments happen automatically:

1. **Push to main** → GitHub Actions triggers
2. **Server changes** (`sabre/**`) → App Runner deploys
3. **UI changes** (`ui/**`) → Amplify builds

### Manual Triggers

Both workflows support manual dispatch:
- GitHub → Actions → Select workflow → "Run workflow"

## Troubleshooting

### App Runner Build Fails

```bash
# Check logs
aws apprunner list-operations \
  --service-arn [ARN] \
  --region us-east-1

# View specific operation
aws apprunner describe-custom-domain \
  --service-arn [ARN] \
  --region us-east-1
```

### Amplify Build Fails

```bash
# List builds
aws amplify list-jobs \
  --app-id $AMPLIFY_APP_ID \
  --branch-name main \
  --region us-east-1

# Get build logs
aws amplify get-job \
  --app-id $AMPLIFY_APP_ID \
  --branch-name main \
  --job-id [JOB_ID] \
  --region us-east-1
```

### GitHub Actions Fails

1. Check Actions tab for error logs
2. Verify all secrets are set correctly
3. Ensure AWS credentials have required permissions

## Cost Estimates

### App Runner
- **Pricing**: $0.007/vCPU-hour + $0.0008/GB-hour
- **1 vCPU, 2 GB**: ~$15/month (always running)
- **Free tier**: 50,000 vCPU-minutes/month

### Amplify
- **Build**: $0.01/build minute
- **Hosting**: $0.15/GB served
- **Free tier**: 1,000 build minutes + 15 GB served/month

### ECR
- **Storage**: $0.10/GB/month
- **Transfer**: $0.09/GB (out to internet)
- **Typical**: <$1/month

## Custom Domains

### Server (App Runner)

```bash
aws apprunner associate-custom-domain \
  --service-arn [ARN] \
  --domain-name api.yourdomain.com \
  --region us-east-1
```

### UI (Amplify)

```bash
aws amplify create-domain-association \
  --app-id $AMPLIFY_APP_ID \
  --domain-name yourdomain.com \
  --sub-domain-settings prefix=www,branchName=main \
  --region us-east-1
```

## Cleanup

### Delete All Resources

```bash
# Delete App Runner service
aws apprunner delete-service --service-arn [ARN] --region us-east-1

# Delete Amplify app
aws amplify delete-app --app-id $AMPLIFY_APP_ID --region us-east-1

# Delete ECR repository
aws ecr delete-repository --repository-name sabre-server --force --region us-east-1
```

## Support

For deployment issues:
- Check GitHub Actions logs
- Review AWS CloudWatch logs
- See [AWS App Runner docs](https://docs.aws.amazon.com/apprunner/)
- See [AWS Amplify docs](https://docs.aws.amazon.com/amplify/)
