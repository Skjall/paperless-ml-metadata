# Setting Up Automated Document Processing with Cron

This guide explains how to set up automatic document processing on a schedule using cron.

## Cron Job Setup

1. Make the `run_auto.sh` script executable:
   ```bash
   chmod +x run_auto.sh
   ```

2. Edit your crontab:
   ```bash
   crontab -e
   ```

3. Add one of the following lines to run the script on your preferred schedule:

   ```
   # Run every day at 2 AM
   0 2 * * * /path/to/paperless-ml-metadata/run_auto.sh >> /path/to/paperless-ml-metadata/cron.log 2>&1

   # Run every Sunday at 3 AM
   0 3 * * 0 /path/to/paperless-ml-metadata/run_auto.sh >> /path/to/paperless-ml-metadata/cron.log 2>&1

   # Run on the 1st and 15th of every month at 4 AM
   0 4 1,15 * * /path/to/paperless-ml-metadata/run_auto.sh >> /path/to/paperless-ml-metadata/cron.log 2>&1
   ```

## Cron Schedule Format

The cron format has five fields:
```
minute hour day month weekday command
```

- **minute**: 0-59
- **hour**: 0-23
- **day**: 1-31
- **month**: 1-12 (or names)
- **weekday**: 0-6 (Sunday is 0 or 7)

## Monitoring

The example above redirects both standard output and errors to a log file. You can check this file to monitor the script's execution:

```bash
tail -f /path/to/paperless-ml-metadata/cron.log
```

## Portainer Setup

If you're using Portainer with the recurring job feature:

1. Go to your Portainer instance
2. Navigate to "Stacks" > select your stack
3. Click on "Add a recurring job"
4. Set your schedule using cron syntax
5. Command: `/bin/bash -c "/path/to/paperless-ml-metadata/run_auto.sh"`
6. Select the host or node where the job should run
7. Save the job