```markdown
# Project Name

This repository contains the code for century Assingment.

## Getting Started

Follow the steps below to set up and run the project locally.

### Prerequisites

- Git
- Conda

### Installation

1. Clone the repository:

   ```bash
   git clone [repository_URL]
   cd [repository_name]
   ```

2. Create a Conda environment:

   ```bash
   conda create -n [name_of_environment] python=3.9
   ```

3. Activate the Conda environment:

   ```bash
   conda activate [name_of_environment]
   ```

4. Download the raw data and store it in the `data/01_raw` directory with the following file names:

   - (`condition.xlsx`) [https://docs.google.com/spreadsheets/d/1yFlaJGa8C6HZRsaptkqUSU7f7ajzMNYr/edit#gid=388865656]
   - (`encounters.parquet`) [https://drive.google.com/file/d/1hZuF_DOm04gVQ_uV0zk8Jg1tmiKyA1cO/view]
   - (`medications.csv`) [https://drive.google.com/file/d/1agKFbBeOtVxpVcvXS_kIexvMBjAAmzYn/view]
   - (`symptoms.csv`) [https://drive.google.com/file/d/1LqgEX5cAW7r6wXaJwqhbCTv-K1BKY0KF/view]
   - (`patients.csv`) [https://drive.google.com/file/d/17bixNERFXqu6G9fMcZjavuGvL3vz6thw/view]
   - (`patient_gender.csv`) [https://drive.google.com/file/d/1dzERlgnsUMWj5UWSKYZFY7jthoqWRXtt/view]

5. Install Kedro:

   ```bash
   pip install kedro
   ```

6. Install project requirements:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the following command to execute the project:

```bash
kedro run
```

The output data will be visible in the `Data/06_models` folder.

## License

This project is licensed under the [License Name] License - see the [LICENSE.md](LICENSE.md) file for details.
```