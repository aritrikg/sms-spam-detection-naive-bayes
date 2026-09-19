# Dataset

The experiment expects a CSV file named `spamraw.csv` with these columns:

- `type`: `ham` or `spam`
- `text`: SMS message text

The manuscript was evaluated on a supplied local CSV containing 5,559 usable rows.
The repository does not redistribute that dataset. Obtain the dataset from the
source permitted by your institution/venue and place `spamraw.csv` in the repository
root before running `experiment.py`.

The paper also distinguishes this 5,559-row local file from the official UCI SMS Spam
Collection record, which reports 5,574 instances.
