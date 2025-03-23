import numpy as np
import pandas as pd
import torch
from typing import Tuple, List

from final_project_models import UniversityMLP, ApplicantMLP
from final_project_envs import UniversityEnvironment


# ===== Utility Functions =====

def calculate_desired_faculty_stats(assigned_faculties, desired_faculties):
    """
    Calculate statistics about how many students got their desired faculty.
    
    Args:
        assigned_faculties: Array of assigned faculty indices
        desired_faculties: Array of desired faculty indices
        
    Returns:
        Tuple of (matches, percentage)
    """
    total_students = len(desired_faculties)
    matches = sum(assigned == desired for assigned, desired in zip(assigned_faculties, desired_faculties))
    percentage = (matches / total_students) * 100
    return matches, percentage


def plot_scenario_comparison(scenarios_data, metric='grade', title=None, figsize=(12, 6)):
    """
    Create bar plots comparing different scenarios by a specific metric.
    
    Args:
        scenarios_data: Dictionary mapping scenario names to their results data
                      Each entry should contain 'mean_grade' and 'desired_percentage'
        metric: Which metric to plot - 'grade' or 'desired'
        title: Optional title for the plot
        figsize: Figure size tuple (width, height)
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    scenario_names = list(scenarios_data.keys())
    
    if metric == 'grade':
        values = [data['mean_grade'] for data in scenarios_data.values()]
        ylabel = 'Mean Grade'
        title = title or 'Comparison of Mean Grades Across Scenarios'
        color = 'skyblue'
    else: 
        values = [data['desired_percentage'] for data in scenarios_data.values()]
        ylabel = 'Students Getting Desired Faculty (%)'
        title = title or 'Percentage of Students Getting Desired Faculty'
        color = 'lightgreen'
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(scenario_names, values, color=color, alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom')
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    return fig


def plot_faculty_distribution(faculty_distributions, scenario_names, n_faculties, figsize=(14, 8)):
    """
    Create bar plots comparing faculty distributions across scenarios.
    
    Args:
        faculty_distributions: List of faculty distribution arrays (from np.bincount)
        scenario_names: List of scenario names corresponding to each distribution
        n_faculties: Number of faculties
        figsize: Figure size tuple (width, height)
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=figsize)
    
    distributions = []
    for dist in faculty_distributions:
        if len(dist) < n_faculties:
            padded_dist = np.pad(dist, (0, n_faculties - len(dist)))
        else:
            padded_dist = dist[:n_faculties]
        distributions.append(padded_dist)
    
    faculty_indices = np.arange(n_faculties)
    width = 0.8 / len(distributions)
    
    for i, (dist, name) in enumerate(zip(distributions, scenario_names)):
        positions = faculty_indices + (i - len(distributions)/2 + 0.5) * width
        bars = ax.bar(positions, dist, width, label=name, alpha=0.7)
    
    ax.set_xlabel('Faculty')
    ax.set_ylabel('Number of Students')
    ax.set_title('Distribution of Students Across Faculties by Scenario')
    ax.set_xticks(faculty_indices)
    ax.set_xticklabels([f'Faculty {i}' for i in faculty_indices])
    ax.legend()
    
    plt.tight_layout()
    return fig


def generate_scenario_comparison_table(scenarios_data):
    """
    Generate a DataFrame comparing key metrics across scenarios.
    
    Args:
        scenarios_data: Dictionary mapping scenario names to their results data
                      Each entry should contain various metrics
        
    Returns:
        pandas DataFrame with scenario comparison
    """
    import pandas as pd
    
    metrics = ['mean_grade', 'desired_percentage', 'matches']
    labels = ['Mean Grade', 'Desired Faculty (%)', 'Number of Matches']
    
    data = {}
    for scenario, results in scenarios_data.items():
        data[scenario] = [results.get(metric, 'N/A') for metric in metrics]
    
    return pd.DataFrame(data, index=labels)


def print_scenario_results(scenario_name, faculties, grades, desired_faculties, additional_metrics=None):
    """
    Print formatted results for a scenario.
    
    Args:
        scenario_name: Name of the scenario to print
        faculties: Array of assigned faculty indices
        grades: Array of final grades
        desired_faculties: Array of desired faculty indices
        additional_metrics: Optional dictionary of additional metrics to print
    """
    matches, percentage = calculate_desired_faculty_stats(faculties, desired_faculties)
    
    print(f"\n{scenario_name} Results:")
    print(f"Mean grade: {np.mean(grades):.2f}")
    print(f"Faculty distribution: {np.bincount(faculties)}")
    print(f"Students who got desired faculty: {matches} ({percentage:.1f}%)")
    
    if additional_metrics:
        for metric_name, metric_value in additional_metrics.items():
            print(f"{metric_name}: {metric_value}")
    
    # Return the calculated metrics
    return {
        'mean_grade': np.mean(grades),
        'faculty_distribution': np.bincount(faculties),
        'matches': matches,
        'desired_percentage': percentage,
        'additional_metrics': additional_metrics
    }


def print_detailed_applicant_results(indices, desired_faculties, assigned_faculties, grades, additional_info=None):
    """
    Print detailed results for specific applicants.
    
    Args:
        indices: Indices of applicants to print details for
        desired_faculties: Array of desired faculty indices
        assigned_faculties: Array of assigned faculty indices
        grades: Array of final grades
        additional_info: Optional dictionary of additional applicant info to print
    """
    print("\nDetailed results for selected applicants:")
    for i in indices:
        desired_faculty = desired_faculties[i]
        print(f"\nApplicant {i}:")
        print(f"Desired faculty: {desired_faculty}")
        print(f"Assigned faculty: {assigned_faculties[i]} {'(DESIRED)' if assigned_faculties[i] == desired_faculty else ''}")
        print(f"Final grade: {grades[i]:.2f}")
        
        if additional_info and i in additional_info:
            for key, value in additional_info[i].items():
                print(f"{key}: {value}")


# ===== Environment Setup Functions =====

def setup_environment(use_university_supplier=False, n_features=10, n_faculties=5):
    """
    Create and set up the university environment.
    
    Args:
        use_university_supplier: Whether to enable the university supplier
        n_features: Number of features for applicants
        n_faculties: Number of faculties in the university
        
    Returns:
        Tuple of (environment, feature column names)
    """
    env = UniversityEnvironment(
        n_features=n_features,
        n_faculties=n_faculties,
        enable_university_supplier=use_university_supplier
    )
    feature_cols = [f"feature_{i}" for i in range(env.n_features)]
    return env, feature_cols


def train_initial_models(env, n_past_applicants=1000, verbose=False):
    """
    Generate past applicants data and train initial models.
    
    Args:
        env: UniversityEnvironment instance
        n_past_applicants: Number of past applicants to generate
        verbose: Whether to print detailed training information
        
    Returns:
        Tuple of (past_data_df, trained_university_model)
    """
    if verbose:
        print("\n=== Generating Past Data and Training Initial Models ===")
    past_df = env.generate_past_applicants(n_past_applicants)
    trained_model = env.train_university_model(past_df, verbose=verbose)
    return past_df, trained_model


def generate_applicants_data(env, n_applicants=1000, verbose=False):
    """
    Generate applicants data for the current iteration.
    
    Args:
        env: UniversityEnvironment instance
        n_applicants: Number of applicants to generate
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (applicants_df, original_features, desired_faculties)
    """
    feature_cols = [f"feature_{i}" for i in range(env.n_features)]
    if verbose:
        print(f"\nGenerating {n_applicants} applicants...")
    applicants_df = env.generate_current_applicants(n_applicants)
    original_features = applicants_df[feature_cols].values
    desired_faculties = applicants_df['desired_faculty'].values
    return applicants_df, original_features, desired_faculties


# ===== Scenario Implementation Functions =====

def get_modified_features(env, applicants_df, applicant_model, use_feature_knowledge=True, track_university_applicants=False, verbose=False):
    """
    Get modified features for all applicants based on their choice of supplier.
    
    Args:
        env: UniversityEnvironment instance
        applicants_df: DataFrame of applicant data
        applicant_model: Trained ApplicantMLP model
        use_feature_knowledge: Whether applicants use knowledge of their features
        track_university_applicants: Whether to track university applicants
        verbose: Whether to print detailed information
        
    Returns:
        Array of modified features
    """
    feature_cols = [f"feature_{i}" for i in range(env.n_features)]
    modified_features = []
    
    if verbose:
        print(f"\nModifying features for {len(applicants_df)} applicants...")
    
    for idx in range(len(applicants_df)):
        student_features = applicants_df.iloc[idx][feature_cols].values
        desired_faculty = applicants_df.iloc[idx]['desired_faculty']
        
        # If not using feature knowledge, start with zero or low-value features
        if use_feature_knowledge:
            base_features = student_features.copy()
        else:
            base_features = np.ones_like(student_features) * 10
        
        applicant_id = idx if track_university_applicants else None
        supp_id, modified_student_features = env.choose_supplier_for_applicant(
            base_features,
            desired_faculty,
            applicant_model,
            applicant_id=applicant_id,
            with_university_supplier=track_university_applicants
        )
        
        # If not using feature knowledge, perform a weighted blend of original features
        # with the supplier modifications to avoid double-counting or extreme values
        if not use_feature_knowledge:
            mod_vector = modified_student_features - base_features
            modified_indices = np.abs(mod_vector) > 1e-5
            
            result_features = np.zeros_like(student_features)
            
            result_features[modified_indices] = modified_student_features[modified_indices]
            
            # For unmodified features, use the original student features
            result_features[~modified_indices] = student_features[~modified_indices]
            
            modified_student_features = np.clip(result_features, 0, 100)
        
        modified_features.append(modified_student_features)
    
    return np.array(modified_features)


def run_no_gaming_scenario(env, model, original_features, desired_faculties, verbose=False):
    """
    Run a scenario where applicants don't modify their features.
    
    Args:
        env: UniversityEnvironment instance
        model: Trained UniversityMLP model
        original_features: Original applicant features
        desired_faculties: Desired faculty for each applicant
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (assigned_faculties, final_grades)
    """
    if verbose:
        print("\n=== Running No Gaming Scenario ===")
    
    assigned_faculties = env.assign_applicants_to_faculties(
        model,
        original_features,
        verbose=verbose
    )
    
    # Calculate final grades
    final_grades = env.recommend(original_features, assigned_faculties)
    
    print_scenario_results(
        "No Gaming", 
        assigned_faculties, 
        final_grades, 
        desired_faculties
    )
    
    return assigned_faculties, final_grades


def run_experiment(verbose=False):
    """
    Run a comprehensive experiment with multiple applicant strategies in a single environment.
    
    This function combines the base scenario, fully exposed example, and university supplier scenario.
    All variations use the same past data and environment setup, with different approaches in
    how applicants choose suppliers in iteration 1.
    
    Args:
        verbose: Whether to print detailed training and execution information
        
    Returns:
        Tuple of (env, feature_cols, scenario_results)
    """
    # University Enabled Environment (we'll use it only for specific variations)
    env, feature_cols = setup_environment(use_university_supplier=True)
    past_df, trained_model = train_initial_models(env, verbose=verbose)
    
    # Generate applicants for iteration 0
    iteration0_applicants_df, iteration0_features, iteration0_desired_faculties = generate_applicants_data(
        env, 
        verbose=verbose
    )
    
    # Run no gaming scenario for iteration 0 - this will be our baseline
    iteration0_faculties, iteration0_grades = run_no_gaming_scenario(
        env, 
        trained_model, 
        iteration0_features, 
        iteration0_desired_faculties,
        verbose=verbose
    )
    
    # Create training data for students from iteration 0
    iteration0_df = pd.DataFrame(iteration0_features, columns=feature_cols)
    iteration0_df['assigned_faculty'] = iteration0_faculties
    iteration0_df['final_grade'] = iteration0_grades
    
    # Iteration 1: Student Learning with Different Strategies
    print("\n=== Iteration 1: Student Learning with Different Strategies ===")
    iteration1_applicants_df, original_features, desired_faculties = generate_applicants_data(
        env,
        verbose=verbose
    )
    
    # Train applicant model based on iteration 0 results
    applicant_model = env.train_applicant_model(iteration0_df, verbose=verbose)
    
    # Collect results for all scenarios
    scenario_results = {}
    
    # ===== Original (No Gaming) Scenario =====
    faculties_original = env.assign_applicants_to_faculties(
        trained_model,
        original_features,
        verbose=verbose
    )
    grades_original = env.recommend(original_features, faculties_original)
    scenario_results["Original (No Gaming)"] = print_scenario_results(
        "Original (No Gaming)", faculties_original, grades_original, desired_faculties
    )
    
    # ===== Modified with Feature Knowledge Scenario =====
    modified_features_with_knowledge = get_modified_features(
        env, 
        iteration1_applicants_df, 
        applicant_model, 
        use_feature_knowledge=True,
        verbose=verbose
    )
    faculties_with_knowledge = env.assign_applicants_to_faculties(
        trained_model,
        modified_features_with_knowledge,
        verbose=verbose
    )
    grades_with_knowledge = env.recommend(original_features, faculties_with_knowledge)
    scenario_results["Modified with Feature Knowledge"] = print_scenario_results(
        "Modified with Feature Knowledge", faculties_with_knowledge, grades_with_knowledge, desired_faculties
    )
    
    # ===== Modified without Feature Knowledge Scenario =====
    modified_features_without_knowledge = get_modified_features(
        env, 
        iteration1_applicants_df, 
        applicant_model, 
        use_feature_knowledge=False,
        verbose=verbose
    )
    faculties_without_knowledge = env.assign_applicants_to_faculties(
        trained_model,
        modified_features_without_knowledge,
        verbose=verbose
    )
    grades_without_knowledge = env.recommend(original_features, faculties_without_knowledge)
    scenario_results["Modified without Feature Knowledge"] = print_scenario_results(
        "Modified without Feature Knowledge", faculties_without_knowledge, grades_without_knowledge, desired_faculties
    )
    
    # ===== With University Reconstruction Scenario =====
    faculties_with_reconstruction = env.assign_applicants_to_faculties_with_reconstruction(
        trained_model,
        modified_features_with_knowledge,
        desired_faculties
    )
    grades_with_reconstruction = env.recommend(original_features, faculties_with_reconstruction)
    scenario_results["With University Reconstruction"] = print_scenario_results(
        "With University Reconstruction", faculties_with_reconstruction, grades_with_reconstruction, desired_faculties
    )
    
    # ===== Fully Exposed Example Scenario =====
    print("\n=== Running Fully Exposed Example Scenario ===")
    modified_features_fully_exposed = []
    # Choose supplier with full knowledge of university model
    for idx in range(len(iteration1_applicants_df)):
        student_features = iteration1_applicants_df.iloc[idx][feature_cols].values
        # Convert desired_faculty to int to ensure proper indexing
        desired_faculty = int(iteration1_applicants_df.iloc[idx]['desired_faculty'])
        supp_id, modified_student_features = env.choose_supplier_for_applicant_fully_exposed(
            student_features,
            desired_faculty,
            trained_model,
        )
        modified_features_fully_exposed.append(modified_student_features)
        if verbose:
            print(f'Student {idx} chose supplier {supp_id}')
    
    modified_features_fully_exposed = np.array(modified_features_fully_exposed)
    faculties_fully_exposed = env.assign_applicants_to_faculties(
        trained_model,
        modified_features_fully_exposed,
        verbose=verbose
    )
    grades_fully_exposed = env.recommend(original_features, faculties_fully_exposed)
    scenario_results["With Full University Model Access"] = print_scenario_results(
        "With Full University Model Access", faculties_fully_exposed, grades_fully_exposed, desired_faculties
    )
    
    # ===== University Supplier Scenario =====
    print("\n=== Running University Supplier Scenario ===")
    # Reset university applicants set
    env.university_applicants = set()
    # Get modified features tracking university applicants
    modified_features_univ_supplier = get_modified_features(
        env,
        iteration1_applicants_df,
        applicant_model,
        track_university_applicants=True,
        verbose=verbose
    )
    # Make assignments with university supplier info
    faculties_univ_supplier = env.assign_applicants_to_faculties(
        trained_model,
        modified_features_univ_supplier,
        desired_faculties,
        np.arange(len(iteration1_applicants_df)),
        verbose=verbose
    )
    grades_univ_supplier = env.recommend(original_features, faculties_univ_supplier)
    additional_metrics = {"Number of students who chose university as supplier": len(env.university_applicants)}
    scenario_results["University Supplier"] = print_scenario_results(
        "University Supplier", faculties_univ_supplier, grades_univ_supplier, desired_faculties, additional_metrics
    )
    
    # Create comparison visualizations for all scenarios
    plot_scenario_comparison(scenario_results, metric='grade', 
                           title='Comparison of Mean Grades Across Scenarios')
    plot_scenario_comparison(scenario_results, metric='desired', 
                           title='Comparison of Students Getting Desired Faculty Across Scenarios')
    
    # Faculty distribution visualization
    faculty_distributions = [results['faculty_distribution'] for results in scenario_results.values()]
    plot_faculty_distribution(faculty_distributions, 
                             list(scenario_results.keys()), 
                             env.n_faculties)
    
    # Generate comparison table
    comparison_table = generate_scenario_comparison_table(scenario_results)
    print("\nScenario Comparison Summary:")
    print(comparison_table)
    
    return env, feature_cols, original_features, {
        "Modified with Knowledge": modified_features_with_knowledge,
        "Modified without Knowledge": modified_features_without_knowledge,
        "Fully Exposed": modified_features_fully_exposed,
        "University Supplier": modified_features_univ_supplier
    }, scenario_results 