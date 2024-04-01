from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_condition, preprocess_encounter, preprocess_observation, preprocess_patient, preprocess_medication , patient_encounter_merge, pe_condition_merge, pec_observation_merge,peco_medication_merge, extract_symptoms

def create_pipeline(**kwargs) -> Pipeline:
    """Create a Kedro pipeline for data preprocessing.

    Returns:
        Pipeline: A Kedro pipeline object representing the data preprocessing workflow.
    """

    return pipeline(
        [
            node(
                func=preprocess_condition,
                inputs="condition",
                outputs="preprocessed_conditions",
                name = "preprocessed_conditions_node",
                # tags=["preprocessing"]
            ),
            node(
                func=preprocess_encounter,
                inputs="encounter",
                outputs="preprocessed_encounters",
                name="preprocessed_encounters_node",
                # tags=["preprocessing"]
            ),
            node(
                func=preprocess_observation,
                inputs="observation",
                outputs="preprocessed_observations",
                name="preprocessed_observations_node",
                # tags=["preprocessing"]
            ),
             node(
                func=extract_symptoms,
                inputs="preprocessed_observations",
                outputs="preprocessed_symptoms",
                name="symptoms_extracting",
                # tags=["preprocessing"]
            ),

            node(
                func=preprocess_patient,
                inputs="patient",
                outputs="preprocessed_patients",
                name="preprocessed_patients_node",
                # tags=["preprocessing"]
            ),
            node(
                func=preprocess_medication,
                inputs="medication",
                outputs="preprocessed_medication",
                name="preprocessed_medication_node",
                # tags=["preprocessing"]
            ),
            node(
                func=patient_encounter_merge,
                inputs=["preprocessed_encounters","preprocessed_patients"],
                outputs="patient_encounter",
                name="patient_encounter_merge",
                # tags=["preprocessing"]
            ),

            node(
                func=pe_condition_merge,
                inputs=['patient_encounter',"preprocessed_conditions"],
                outputs="pe_condition",
                name="patient_encounter_condition_merge",
                # tags=["preprocessing"]
            ),

            node(
                func=pec_observation_merge,
                inputs=['pe_condition',"preprocessed_symptoms"],
                outputs="pec_observation",
                name="patient_encounter_condition_observation_merge",
                # tags=["preprocessing"]
            ),

            node(
                func=peco_medication_merge,
                inputs=['pec_observation',"preprocessed_medication"],
                outputs="model_input",
                name="model_input"
            ),
        ]
    )
