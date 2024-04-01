import pandas as pd
import re

def _to_lower(x: pd.Series) -> pd.Series:
    """
    Converts the values in a pandas Series to lowercase if they are strings.

    Args:
        x (pd.Series): The pandas Series to convert to lowercase.

    Returns:
        pd.Series: The pandas Series with lowercase values (for string values only).
    """
    if x.dtype == 'object':
        return x.str.lower()
    else:
        return x

def _to_datetime(x: pd.Series) -> pd.Series:
    """
    Converts a pandas Series to datetime. Replaces invalid dates with NaT.

    Args:
        x (pd.Series): The pandas Series to convert to datetime.

    Returns:
        pd.Series: The converted pandas Series with datetime values.
    """
    # Function to check if a value is a valid date
    def is_valid_date(value):
        try:
            pd.to_datetime(value)
            return True
        except (ValueError, TypeError):
            return False
    
    # Convert to datetime
    x = pd.to_datetime(x, errors='coerce')
    
    # Apply additional checks
    for i in range(len(x)):
        if x.iloc[i] is not pd.NaT:
            date = x.iloc[i].date()
            if not (1 <= date.month <= 12) or date.day > 31:
                x.iloc[i] = pd.NaT

    return x

def _resolve_status(stop_time):
    """
    Determine the status of a task based on its stop time.

    Parameters:
    - stop_time (pandas.Timestamp or None): The stop time of the task. If None or NaN, the task is considered ongoing.

    Returns:
    - str: The status of the task. If stop_time is None or NaN, returns "ongoing"; otherwise, returns "resolved".
    """
    if pd.isnull(stop_time):
        return "ongoing"
    else:
        return "resolved"
    


def _extract_mode(prescription):

    """
    Extract the mode of administration from a prescription string.

    Parameters:
    - prescription (str): A string containing information about the prescription.

    Returns:
    - str: The mode of administration extracted from the prescription. If 'oral' is found in the prescription,
      returns 'oral'. If 'injectable' is found in the prescription, returns 'injectable'. Otherwise, returns 'Unknown'.
    """

    if 'oral' in prescription:
        return 'oral'
    elif 'injectable' in prescription:
        return 'injectable'
    else:
        return 'Unknown'
    

def _input_layer_mapper(df: pd.DataFrame,column_mapping: list[dict[str, str]]) -> pd.DataFrame:

    """
    Map columns from an input DataFrame to a new DataFrame based on the specified column mapping.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing columns to be mapped.
    - column_mapping (list[dict[str, str]]): A list of dictionaries where each dictionary specifies the mapping
      between columns in the input DataFrame and columns in the output DataFrame. Each dictionary should have
      two key-value pairs: 'input' represents the column name in the input DataFrame, and 'output' represents
      the corresponding column name in the output DataFrame. If the value is "missing", the corresponding column
      in the output DataFrame will be filled with None.

    Returns:
    - pd.DataFrame: A new DataFrame where columns are mapped according to the provided column mapping. If a column
      in the column_mapping list has a value of "missing", the corresponding column in the output DataFrame will
      be filled with None.

    Example:
    Suppose df is:
    |  A  |  B  |  C  |
    |-----|-----|-----|
    |  1  |  2  |  3  |
    |  4  |  5  |  6  |

    And column_mapping is:
    [{'input': 'A', 'output': 'X'}, {'input': 'C', 'output': 'Y'}, {'input': 'B', 'output': 'missing'}]

    Then, the output DataFrame will be:
    |  X  |   Y   |   missing  |
    |-----|-------|------------|
    |  1  |   3   |    None    |
    |  4  |   6   |    None    |
    """
        # create  a dataframe
    new_df = pd.DataFrame(columns=column_mapping.keys())
    for col,val in column_mapping.items():
        new_df[col] = df[val] if val != "missing" else None

    return new_df



def _extract_dosage(prescription):

    """
    Extract the dosage information from a prescription string using regular expression.

    Parameters:
    - prescription (str): A string containing information about the prescription.

    Returns:
    - str or None: The extracted dosage information from the prescription if found, otherwise returns None.

    Example:
    Suppose prescription is "Take 1 tablet, 3 times a day, 200 mg each".
    The output will be "200 mg".
    """

    # Use regular expression to find dosage pattern
    dosage_pattern = r'(\d+\s*(?:mg|mg\/ml))'
    match = re.search(dosage_pattern, prescription)
    if match:
        return match.group(1)
    else:
        return None
    

def _is_datetime_not_naT(column):
    """
    Converts a DataFrame column to a boolean mask where True corresponds to non-NaT values
    and False corresponds to NaT values.

    Args:
        column (pd.Series): The DataFrame column to process.

    Returns:
        pd.Series: A boolean mask indicating whether each value in the column is not NaT.
    """
    return ~column.isna()

def _split_symptoms_column(df:pd.DataFrame, column_name:list[str]):
    # Extract symptom names and values from the given column
    symp = []
    for entry in df[column_name].iterrows():
        parts = entry[1]['symptoms'].split(';')
        temp_dict = {x.split(':')[0]:x.split(':')[1] for x in parts}
        temp_dict['patient_id'] = entry[1]['patient_id']
        symp.append(temp_dict)
    # Create a new DataFrame with columns for each symptom
    new_df = pd.DataFrame(symp)
    return new_df

def drop_columnName(column_name):
    """
    Identify and drop columns with suffix '_y', and rename columns with suffix '_x' by removing the suffix.

    Parameters:
    - column_name (list[str]): A list of column names.

    Returns:
    - tuple[list[str], dict[str, str]]: A tuple containing two elements:
        - A list of columns to be dropped, identified by having the suffix '_y'.
        - A dictionary mapping columns with the suffix '_x' to their new names, obtained by removing the suffix.

    Example:
    Suppose column_name is ['A_x', 'B_y', 'C_x', 'D_y'].
    The output will be (['B_y'], {'A_x': 'A', 'C_x': 'C'}).
    """
    drop_column = [x for x in column_name if '_y' in x]
    new_name = {x:x.split('_')[0] for x in column_name if '_x' in x}

    return drop_column, new_name

# Function to check if value is NaT or not
# function for the condition dataset
def preprocess_condition(condition: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data for condition.

    Args:
        condition (pd.DataFrame): The input DataFrame containing condition data.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with the following modifications:
            - 'START' and 'STOP' columns are converted to datetime objects.
            - 'PATIENT', 'ENCOUNTER', and 'DESCRIPTION' columns are converted to lowercase.
            - 'condition_id' column is created by concatenating 'patient_id', 'encounter_id', and 'source_code'.
            - 'status' column is determined based on the 'resolved_date', where if the value is NaT, it is set to 'ongoing', otherwise 'resolved'.

    Note:
        Ensure that the functions `_to_datetime`, `_to_lower`, and `_input_layer_mapper` are defined elsewhere in your codebase.

    Example:
        # Preprocess condition DataFrame
        preprocessed_condition = preprocess_condition(condition_df)
    """
    condition_input_layer_map = {
        "condition_id": "missing",
        "patient_id": "PATIENT", 
        "encounter_id": "ENCOUNTER", 
        "claim_id": "missing",
        "recorded_date": "START", 
        "onset_date": "START", 
        "resolved_date": "STOP", 
        "status": "missing",
        "condition_type": "missing", 
        "source_code_type": "missing", 
        "source_code": "CODE",
        "source_description": "DESCRIPTION", 
        "normalize_code": "missing", 
        "normalized_description": "missing",
        "condition_rank": "missing", 
        "present_on_admit_code": "missing", 
        "present_on_admit_description": "missing",
        "data_source": "missing",
    }


    condition['START'] = _to_datetime(condition['START'])
    condition['STOP'] = _to_datetime(condition['STOP'])
    for x in ['PATIENT','ENCOUNTER','DESCRIPTION']:
        condition[x] = _to_lower(condition[x])

    condition = _input_layer_mapper(condition, condition_input_layer_map)
    condition['condition_id'] = condition["patient_id"].astype(str) + "_" + condition["encounter_id"].astype(str) + "_" + condition["source_code"].astype(str)

    condition['status'] = condition['resolved_date'].apply(_resolve_status)

    return condition


# function for teh encounter dataset

def preprocess_encounter(encounter:pd.DataFrame) -> pd.DataFrame:

    """
    Preprocesses the data for encounter.

    Args:
        encounter (pd.DataFrame): The input DataFrame containing encounter data.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with the following modifications:
            - Column names are mapped to a standardized format using the provided mapping.
            - 'START' and 'STOP' columns are converted to datetime objects.
            - 'LENGTH' column is calculated as the difference between 'STOP' and 'START' columns.
            - 'PAID_AMOUNT' column is calculated as the difference between 'TOTAL_CLAIM_COST' and 'PAYER_COVERAGE' columns.
            - 'ALLOWED' column is calculated as the difference between 'TOTAL_CLAIM_COST' and 'PAYER_COVERAGE' columns.
            - 'BASE_ENCOUNTER_COST' column is copied from the original 'BASE_ENCOUNTER_COST' column.
            - NaN values in 'REASONCODE' column are replaced with zeros.
            - 'REASONCODE' column is converted to integer type.

    Note:
        Ensure that the function `_to_datetime`, `_to_lower`, and `_input_layer_mapper` are defined elsewhere in your codebase.

    Example:
        # Preprocess encounter DataFrame
        preprocessed_encounter = preprocess_encounter(encounter_df)
    """

    column_names_mapping = {
    'encounter_id': 'Id',
    'patient_id': 'PATIENT',
    'encounter_type': 'ENCOUNTERCLASS',
    'encounter_start_date': 'START',
    'encounter_end_date': 'STOP',
    'length_of_stay': 'LENGTH',
    'admit_source_code': 'missing',
    'admit_source_description': 'missing',
    'admit_type_code': 'CODE',
    'admit_type_description': 'DESCRIPTION',
    'discharge_disposition_code': 'missing',
    'discharge_disposition_description': 'missing',
    'attending_provider_id': 'PROVIDER',
    'facility_npi': 'missing',
    'primary_diagnosis_code_type': 'REASONCODE',
    'primary_diagnosis_description': 'REASONDESCRIPTION',
    'ms_drg_code': 'missing',
    'ms_drg_description': 'missing',
    'apr_drg_code': 'missing',
    'apr_drg_description': 'missing',
    'paid_amount': 'PAID_AMOUNT',
    'allowed_amount': 'ALLOWED',
    'charge_amount': 'BASE_ENCOUNTER_COST',
    'data_source': 'missing',
    'organization': 'ORGANIZATION',
    'payer' : "PAYER",
    'payer_coverage':"PAYER_COVERAGE",

}
    
    encounter['START'] = _to_datetime(encounter['START'])
    encounter['STOP'] = _to_datetime(encounter['STOP'])
    encounter['LENGTH'] = encounter['STOP']-encounter['START']
    encounter["PAID_AMOUNT"] = encounter['TOTAL_CLAIM_COST']-encounter['PAYER_COVERAGE']
    encounter["ALLOWED"] = encounter['TOTAL_CLAIM_COST']-encounter['PAYER_COVERAGE']
    encounter["BASE_ENCOUNTER_COST"] = encounter['BASE_ENCOUNTER_COST']
    # Replace NaN values with zeros in the 'REASONCODE' column
    encounter['REASONCODE'] = encounter['REASONCODE'].fillna(0)
    encounter["BASE_ENCOUNTER_COST"] = encounter['REASONCODE'].astype(int)

    for x in ['Id','PATIENT','ENCOUNTERCLASS','DESCRIPTION','PROVIDER','REASONDESCRIPTION','PAYER','ORGANIZATION']:
        encounter[x] = _to_lower(encounter[x])

    encounter = _input_layer_mapper(encounter, column_names_mapping)
    return encounter



def preprocess_observation(observation:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data for observation.

    Args:
        observation (pd.DataFrame): The input DataFrame containing observation data.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with the following modifications:
            - Column names are mapped to a standardized format using the provided mapping.
            - 'PATIENT', 'PATHOLOGY', and 'SYMPTOMS' columns are converted to lowercase.
            - 'observation_id' column is created by concatenating 'oid' and 'patient_id'.
    
    Note:
        Ensure that the function `_to_lower` and `_input_layer_mapper` are defined elsewhere in your codebase.

    Example:
        # Preprocess observation DataFrame
        preprocessed_observation = preprocess_observation(observation_df)
    """
    columns_dict = {
    'observation_id': 'missing',
    'patient_id': 'PATIENT',
    'encounter_id': 'missing',
    'panel_id': 'missing',
    'observation_date': 'missing',
    'observation_type': 'missing',
    'source_code_type': 'missing',
    'source_code': 'missing',
    'source_description': 'missing',
    'normalized_code_type': 'missing',
    'normalized_code': 'missing',
    'normalized_description': 'missing',
    'result': 'PATHOLOGY',
    'source_units': 'missing',
    'normalized_units': 'missing',
    'source_reference_range_low': 'missing',
    'source_reference_range_high': 'missing',
    'normalized_reference_range_low': 'missing',
    'normalized_reference_range_high': 'missing',
    'data_source': 'missing',
    'num_symptoms':'NUM_SYMPTOMS',
    'symptoms':'SYMPTOMS',
}
    for x in ['PATIENT','PATHOLOGY','SYMPTOMS']:
        observation[x] = _to_lower(observation[x])
    
    observation = _input_layer_mapper(observation, columns_dict)
    observation['observation_id'] = 'oid'+ '_' + observation['patient_id']
    return observation

# function for the patient dataseq
def preprocess_patient(patient:pd.DataFrame) -> pd.DataFrame:

    column_dict= {
    'patient_id': 'PATIENT_ID',
    'first_name': 'FIRST',
    'last_name': 'LAST',
    'sex': 'GENDER',
    'race': 'RACE',
    'birth_date': 'BIRTHDATE',
    'death_date': 'DEATHDATE',
    'death_flag': 'DEATHFLAG',
    'address': 'ADDRESS',
    'city': 'CITY',
    'state': 'STATE',
    'zip_code': 'ZIP',
    'county': 'COUNTY',
    'latitude': 'LAT',
    'longitude': 'LON',
    'data_source': 'missing',
    'healthcare_expenses':'HEALTHCARE_EXPENSES',
    'healthcare_coverage':'HEALTHCARE_COVERAGE',
    'passport':'PASSPORT',
    'drivers':'DRIVERS',
    'ssn':'SSN'

}

    patient['BIRTHDATE'] = _to_datetime(patient['BIRTHDATE'])
    patient['DEATHDATE'] = _to_datetime(patient['DEATHDATE'])
    patient['DEATHFLAG'] = _is_datetime_not_naT(patient['DEATHDATE'])

    for x in ['PATIENT_ID','FIRST','LAST','GENDER','RACE','ADDRESS','CITY','STATE','COUNTY','PASSPORT','DRIVERS','SSN']:
        patient[x] = _to_lower(patient[x])

    patient = _input_layer_mapper(patient, column_dict)
    return patient


# function for the medication dataset
def preprocess_medication(medication:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data for medication.

    Args:
        medication (pd.DataFrame): The input DataFrame containing medication data.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with the following modifications:
            - Column names are mapped to a standardized format using the provided mapping.
            - 'START' and 'STOP' columns are converted to datetime objects.
            - 'DAYS' column is calculated as the difference between 'STOP' and 'START' columns.
            - NaN values in 'CODE' column are replaced with zeros.
            - 'DESCRIPTION', 'PATIENT', 'ENCOUNTER', and 'PAYER' columns are converted to lowercase.
            - 'ROUTE' column is determined using the '_extract_mode' function applied to the 'DESCRIPTION' column.
            - 'STRENGTH' column is determined using the '_extract_dosage' function applied to the 'DESCRIPTION' column.

    Note:
        Ensure that the functions `_to_datetime`, `_to_lower`, `_extract_mode`, `_extract_dosage`, and `_input_layer_mapper`
        are defined elsewhere in your codebase.

    Example:
        # Preprocess medication DataFrame
        preprocessed_medication = preprocess_medication(medication_df)
    """
    column_names = {
    'medication_id': 'ENCOUNTER',
    'patient_id': 'PATIENT',
    'encounter_id': 'ENCOUNTER',
    'dispensing_date': 'STOP',
    'prescribing_date': 'START',
    'source_code_type': 'missing',
    'source_code': 'missing',
    'source_description': 'missing',
    'ndc_code': 'missing',
    'ndc_description': 'missing',
    'rxnorm_code': 'CODE',
    'rxnorm_description': 'DESCRIPTION',
    'atc_code': 'missing',
    'atc_description': 'missing',
    'route': 'ROUTE',
    'strength': 'STRENGTH',
    'quantity': 'missing',
    'quantity_unit': 'missing',
    'days_supply': 'DAYS',
    'practitioner_id': 'missing',
    'data_source': 'missing',
    'payer' : "PAYER",
    'payer_coverage':"PAYER_COVERAGE",
}

    medication['START'] = _to_datetime(medication['START'])
    medication['STOP'] = _to_datetime(medication['STOP'])
    medication['DAYS'] = medication['STOP']-medication['START']
    medication['CODE'] = medication['CODE'].fillna(0)


    for x in ['DESCRIPTION',"PATIENT","ENCOUNTER",'PAYER']:
        medication[x] = _to_lower(medication[x])

    medication['ROUTE'] = medication['DESCRIPTION'].apply(_extract_mode)
    medication['STRENGTH'] = medication['DESCRIPTION'].apply(_extract_dosage)


    medication = _input_layer_mapper(medication, column_names)

    return medication

def extract_symptoms(preprocess_observation:pd.DataFrame) -> pd.DataFrame:
    """Extracts the symptoms from the observation data."""

    symptoms_df = _split_symptoms_column(preprocess_observation,['patient_id','symptoms'])
    preprocess_observation = preprocess_observation.merge(symptoms_df,on='patient_id')
    return preprocess_observation

def patient_encounter_merge(
    preprocessed_encounters: pd.DataFrame, preprocessed_patients: pd.DataFrame) -> pd.DataFrame:
    """Combines patient and encounter data to create a model input table.

    Args:
        preprocessed_encounters (pd.DataFrame): Preprocessed data for encounters.
        preprocessed_patients (pd.DataFrame): Preprocessed data for patients.

    Returns:
        pd.DataFrame: Model input table obtained by merging preprocessed patient and encounter data.

    Notes:
        The function merges the preprocessed patient and encounter dataframes based on the 'patient_id' column.
        It then drops columns ending with '_y' and renames columns ending with '_x' by removing the suffix '_x'.

    Example:
        # Merge preprocessed patient and encounter data
        model_input_table = patient_encounter_merge(preprocessed_encounters, preprocessed_patients)
    """

    preprocessed_patients = preprocessed_patients.merge(preprocessed_encounters, on="patient_id", how="left")

    drop_column,new_name = drop_columnName(preprocessed_patients.columns)
    preprocessed_patients = preprocessed_patients.drop(drop_column, axis=1)
    preprocessed_patients = preprocessed_patients.rename(columns=new_name)
    return preprocessed_patients

def pe_condition_merge(
    patient_encounter: pd.DataFrame, preprocessed_conditions: pd.DataFrame
) -> pd.DataFrame:
    """Combines patient encounter and condition data to create a model input table.

    Args:
        patient_encounter (pd.DataFrame): Preprocessed data for patient encounters.
        preprocessed_conditions (pd.DataFrame): Preprocessed data for conditions.

    Returns:
        pd.DataFrame: Model input table obtained by merging preprocessed patient encounter and condition data.

    Notes:
        The function merges the preprocessed patient encounter and condition dataframes based on the 'patient_id' column.
        It then drops columns ending with '_y' and renames columns ending with '_x' by removing the suffix '_x'.

    Example:
        # Merge preprocessed patient encounter and condition data
        model_input_table = pe_condition_merge(patient_encounter, preprocessed_conditions)
    """

    patient_encounter = patient_encounter.merge(preprocessed_conditions, on="patient_id", how="left")
    drop_column,new_name = drop_columnName(patient_encounter.columns)

    patient_encounter = patient_encounter.drop(drop_column, axis=1)
    patient_encounter = patient_encounter.rename(new_name)
    return patient_encounter


def pec_observation_merge(
    pe_condition: pd.DataFrame, preprocessed_observations: pd.DataFrame
) -> pd.DataFrame:
    """Combines patient encounter condition and observation data to create a model input table.

    Args:
        pe_condition (pd.DataFrame): Preprocessed data for patient encounter conditions.
        preprocessed_observations (pd.DataFrame): Preprocessed data for observations.

    Returns:
        pd.DataFrame: Model input table obtained by merging preprocessed patient encounter condition and observation data.

    Notes:
        The function merges the preprocessed patient encounter condition and observation dataframes based on the 'patient_id' column.
        It then drops columns ending with '_y' and renames columns ending with '_x' by removing the suffix '_x'.

    Example:
        # Merge preprocessed patient encounter condition and observation data
        model_input_table = pec_observation_merge(pe_condition, preprocessed_observations)
    """

    pe_condition = pe_condition.merge(preprocessed_observations, on="patient_id", how="left")
    drop_column,new_name = drop_columnName(pe_condition.columns)
    pe_condition = pe_condition.drop(drop_column, axis=1)
    pe_condition = pe_condition.rename(new_name)
    return pe_condition


def peco_medication_merge(
    pec_observation: pd.DataFrame, preprocessed_medication: pd.DataFrame
) -> pd.DataFrame:
    """Combines patient encounter condition observation and medication data to create a model input table.

    Args:
        pec_observation (pd.DataFrame): Preprocessed data for patient encounter condition observations.
        preprocessed_medication (pd.DataFrame): Preprocessed data for medications.

    Returns:
        pd.DataFrame: Model input table obtained by merging preprocessed patient encounter condition observation and medication data.

    Notes:
        The function merges the preprocessed patient encounter condition observation and medication dataframes based on the 'patient_id' column.
        It then drops columns ending with '_y' and renames columns ending with '_x' by removing the suffix '_x'.

    Example:
        # Merge preprocessed patient encounter condition observation and medication data
        model_input_table = peco_medication_merge(pec_observation, preprocessed_medication)
    """
    custom_suffixes = ('_pec', '_medication')
    pec_observation = pec_observation.merge(preprocessed_medication, on="patient_id",suffixes=custom_suffixes)
    drop_column,new_name = drop_columnName(pec_observation.columns)
    pec_observation = pec_observation.drop(drop_column, axis=1)
    pec_observation = pec_observation.rename(new_name)
    return pec_observation