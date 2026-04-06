"""
Enhanced Data Validation Module for Proteomics Analysis Toolkit

Functions for validating metadata/data file consistency and providing 
interpretable error messages when samples or controls are missing.
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional


class SampleMatchingError(Exception):
    """Custom exception for sample matching issues."""
    def __init__(self, message):
        super().__init__(message)


class ControlSampleError(Exception):
    """Custom exception for control sample issues."""
    def __init__(self, message):
        super().__init__(message)


def validate_metadata_data_consistency(
    metadata: pd.DataFrame,
    metadata_sample_names: List[str],
    protein_columns: List[str],
    control_column: str,
    control_labels: List[str],
    verbose: bool = True
) -> Dict:
    """
    Validate consistency between metadata and protein data files.
    
    Parameters:
    -----------
    metadata : pd.DataFrame
        Sample metadata
    protein_data : pd.DataFrame
        Protein quantitation data
    metadata_sample_names : List[str]
        Sample names from metadata (first column)
    protein_columns : List[str] 
        Column names from protein data
    control_column : str
        Column name containing control sample designations
    control_labels : List[str]
        Labels identifying control samples
    verbose : bool, default True
        Whether to print detailed validation results
        
    Returns:
    --------
    Dict containing validation results and diagnostic information
    """
    
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'diagnostics': {}
    }
    
    if verbose:
        print("METADATA/DATA CONSISTENCY VALIDATION")
        print("=" * 50)
    
    # 1. Find which metadata samples have corresponding data columns
    found_samples = []
    missing_samples = []
    
    for sample_name in metadata_sample_names:
        found = False
        for col in protein_columns:
            if sample_name in col or col.startswith(sample_name):
                found_samples.append(sample_name)
                found = True
                break
        if not found:
            missing_samples.append(sample_name)
    
    # 2. Identify control samples in metadata
    control_samples_in_metadata = []
    if control_column in metadata.columns:
        for _, row in metadata.iterrows():
            sample_name = row.iloc[0]  # First column is sample name
            control_value = row[control_column]
            if control_value in control_labels:
                control_samples_in_metadata.append((sample_name, control_value))
    else:
        error_msg = f"Control column '{control_column}' not found in metadata"
        results['errors'].append(error_msg)
        results['is_valid'] = False
    
    # 3. Check if control samples are found in data
    missing_control_samples = []
    found_control_samples = []
    
    for sample_name, control_type in control_samples_in_metadata:
        if sample_name in found_samples:
            found_control_samples.append((sample_name, control_type))
        else:
            missing_control_samples.append((sample_name, control_type))
    
    # 4. Generate errors and warnings
    if missing_samples:
        error_msg = (f"Found {len(missing_samples)} samples in metadata that have no corresponding "
                    f"columns in protein data: {missing_samples[:5]}{'...' if len(missing_samples) > 5 else ''}")
        results['errors'].append(error_msg)
        results['is_valid'] = False
        
    if missing_control_samples:
        control_types = set(ct for _, ct in missing_control_samples)
        error_msg = (f"Found {len(missing_control_samples)} control samples in metadata that are missing "
                    f"from protein data. Control types affected: {list(control_types)}. "
                    f"Missing samples: {[s for s, _ in missing_control_samples[:3]]}{'...' if len(missing_control_samples) > 3 else ''}")
        results['errors'].append(error_msg)
        results['is_valid'] = False
    
    # 5. Store diagnostics
    results['diagnostics'] = {
        'total_metadata_samples': len(metadata_sample_names),
        'samples_found_in_data': len(found_samples),
        'samples_missing_from_data': len(missing_samples),
        'total_control_samples_in_metadata': len(control_samples_in_metadata),
        'control_samples_found_in_data': len(found_control_samples),
        'control_samples_missing_from_data': len(missing_control_samples),
        'found_samples': found_samples,
        'missing_samples': missing_samples,
        'control_samples_in_metadata': control_samples_in_metadata,
        'found_control_samples': found_control_samples,
        'missing_control_samples': missing_control_samples
    }
    
    # 6. Print validation summary if verbose
    if verbose:
        diag = results['diagnostics']
        print(f"Metadata samples: {diag['total_metadata_samples']}")
        print(f"  Found in protein data: {diag['samples_found_in_data']}")
        print(f"  Missing from protein data: {diag['samples_missing_from_data']}")
        print(f"\nControl samples in metadata: {diag['total_control_samples_in_metadata']}")
        print(f"  Found in protein data: {diag['control_samples_found_in_data']}")
        print(f"  Missing from protein data: {diag['control_samples_missing_from_data']}")
        
        if diag['missing_control_samples']:
            print("\nMissing control samples by type:")
            missing_by_type = {}
            for sample, ctrl_type in diag['missing_control_samples']:
                if ctrl_type not in missing_by_type:
                    missing_by_type[ctrl_type] = []
                missing_by_type[ctrl_type].append(sample)
            for ctrl_type, samples in missing_by_type.items():
                print(f"  {ctrl_type}: {samples}")
        
        if results['errors']:
            print("\nVALIDATION FAILED")
            for error in results['errors']:
                print(f"  ERROR: {error}")
        else:
            print("\nVALIDATION PASSED")
            
    return results


def enhanced_sample_processing(
    metadata: pd.DataFrame,
    protein_data: pd.DataFrame,
    group_column: str,
    group_labels: List[str],
    control_column: str,
    control_labels: List[str],
    toolkit_module,  # ptk module
    strict_validation: bool = True
) -> Tuple[List[str], Dict, Dict, List[str], List[str], Dict]:
    """
    Enhanced sample processing with comprehensive validation and error handling.
    
    Parameters:
    -----------
    metadata : pd.DataFrame
        Sample metadata
    protein_data : pd.DataFrame  
        Protein quantitation data
    group_column : str
        Column for study group classification
    group_labels : List[str]
        Study group labels
    control_column : str
        Column for control identification  
    control_labels : List[str]
        Control sample labels
    toolkit_module : module
        Proteomics toolkit module (ptk)
    strict_validation : bool, default True
        Whether to raise errors on validation failures
        
    Returns:
    --------
    Tuple of (final_sample_columns, sample_metadata, group_distribution, 
              control_samples, study_samples, group_colors)
    """
    
    # Step 1: Extract sample information
    metadata_sample_names = metadata.iloc[:, 0].tolist()
    protein_columns = protein_data.columns.tolist()
    
    # Step 2: Validate metadata/data consistency
    validation_results = validate_metadata_data_consistency(
        metadata=metadata,
        metadata_sample_names=metadata_sample_names,
        protein_columns=protein_columns,
        control_column=control_column,
        control_labels=control_labels,
        verbose=True
    )
    
    # Step 3: Handle validation failures
    if not validation_results['is_valid']:
        error_summary = "\n".join(validation_results['errors'])
        if strict_validation:
            raise SampleMatchingError(f"Sample matching validation failed:\n{error_summary}")
        else:
            print("WARNING: Validation issues detected but continuing with available data:")
            for error in validation_results['errors']:
                print(f"   {error}")
    
    # Step 4: Proceed with standard sample processing
    sample_columns = []
    for col in protein_columns:
        if any(sample_name in col or col.startswith(sample_name) for sample_name in metadata_sample_names):
            sample_columns.append(col)
    
    # Clean sample names
    cleaned_sample_names_dict = toolkit_module.data_import.clean_sample_names(
        sample_columns,
        auto_detect=True
    )
    
    final_sample_columns = list(cleaned_sample_names_dict.values())
    
    # Create sample metadata mapping
    sample_metadata = {}
    original_sample_names = metadata.iloc[:, 0].tolist()
    
    for i, original_name in enumerate(original_sample_names):
        if original_name.startswith('Total-PT'):
            core_part = original_name[8:]
            for cleaned_name in final_sample_columns:
                if core_part in cleaned_name:
                    row_data = metadata.iloc[i].to_dict()
                    metadata_dict = {k: v for k, v in row_data.items() if k != metadata.columns[0]}
                    sample_metadata[cleaned_name] = metadata_dict
                    break
    
    # Step 5: Classify samples with additional validation
    group_distribution, control_samples, study_samples, sample_metadata, group_colors = toolkit_module.classify_samples(
        sample_metadata=sample_metadata,
        group_column=group_column,
        group_labels=group_labels,
        control_column=control_column,
        control_labels=control_labels,
        apply_systematic_colors=True
    )
    
    # Step 6: Final validation of control sample detection
    expected_control_count = validation_results['diagnostics']['control_samples_found_in_data']
    actual_control_count = len(control_samples)
    
    if expected_control_count != actual_control_count:
        warning_msg = (f"Expected {expected_control_count} control samples to be classified, "
                      f"but only {actual_control_count} were found. Check sample name matching logic.")
        if strict_validation:
            raise ControlSampleError(warning_msg)
        else:
            print(f"WARNING: {warning_msg}")
    
    print(f"Done: Enhanced sample processing complete: {len(final_sample_columns)} samples, {len(sample_metadata)} with metadata")
    
    return final_sample_columns, sample_metadata, group_distribution, control_samples, study_samples, group_colors


def generate_sample_matching_diagnostic_report(
    validation_results: Dict,
    output_file: Optional[str] = None
) -> str:
    """
    Generate a detailed diagnostic report for sample matching issues.
    
    Parameters:
    -----------
    validation_results : Dict
        Results from validate_metadata_data_consistency
    output_file : str, optional
        Path to save the report
        
    Returns:
    --------
    str: Formatted diagnostic report
    """
    
    diag = validation_results['diagnostics']
    
    report = []
    report.append("SAMPLE MATCHING DIAGNOSTIC REPORT")
    report.append("=" * 50)
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary statistics
    report.append("SUMMARY STATISTICS")
    report.append("-" * 30)
    report.append(f"Total samples in metadata: {diag['total_metadata_samples']}")
    report.append(f"Samples found in protein data: {diag['samples_found_in_data']}")
    report.append(f"Samples missing from protein data: {diag['samples_missing_from_data']}")
    report.append(f"Match rate: {diag['samples_found_in_data']/diag['total_metadata_samples']*100:.1f}%")
    report.append("")
    
    # Control sample analysis
    report.append("CONTROL SAMPLE ANALYSIS")
    report.append("-" * 30)
    report.append(f"Control samples in metadata: {diag['total_control_samples_in_metadata']}")
    report.append(f"Control samples found in data: {diag['control_samples_found_in_data']}")
    report.append(f"Control samples missing: {diag['control_samples_missing_from_data']}")
    if diag['total_control_samples_in_metadata'] > 0:
        control_match_rate = diag['control_samples_found_in_data'] / diag['total_control_samples_in_metadata'] * 100
        report.append(f"Control match rate: {control_match_rate:.1f}%")
    report.append("")
    
    # Missing samples details
    if diag['missing_samples']:
        report.append("MISSING SAMPLES DETAILS")
        report.append("-" * 30)
        for i, sample in enumerate(diag['missing_samples'][:10]):  # Show first 10
            report.append(f"{i+1}. {sample}")
        if len(diag['missing_samples']) > 10:
            report.append(f"... and {len(diag['missing_samples']) - 10} more")
        report.append("")
    
    # Missing control samples details
    if diag['missing_control_samples']:
        report.append("MISSING CONTROL SAMPLES DETAILS")
        report.append("-" * 30)
        missing_by_type = {}
        for sample, ctrl_type in diag['missing_control_samples']:
            if ctrl_type not in missing_by_type:
                missing_by_type[ctrl_type] = []
            missing_by_type[ctrl_type].append(sample)
            
        for ctrl_type, samples in missing_by_type.items():
            report.append(f"{ctrl_type}: {samples}")
        report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 30)
    if diag['samples_missing_from_data'] > 0:
        report.append("1. Check if sample names in metadata match column names in protein data")
        report.append("2. Verify that protein data file contains all expected samples")
        report.append("3. Check for naming inconsistencies (e.g., EISAIPool vs PlatePool)")
    if diag['control_samples_missing_from_data'] > 0:
        report.append("4. Ensure control samples are present in protein quantitation data")
        report.append("5. Verify control sample naming consistency across files")
    if validation_results['is_valid']:
        report.append("All samples successfully matched - no action needed")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Diagnostic report saved to: {output_file}")
    
    return report_text
