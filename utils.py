import numpy as np
from BostonMech import BostonMechanism
from DA import DeferredAcceptance
import random


def get_rank_at_school(which_school, which_student, school_pref_mat):
    return np.where(school_pref_mat[which_school] == which_student)[0][0]


def safe_if_first_choice(
    which_school, which_student, school_pref_mat, student_pref_mat, school_capacities
):
    competitors = np.where(student_pref_mat[:, 0] == which_school)
    # get this student's rank at this school
    this_student_rank = get_rank_at_school(which_school, which_student, school_pref_mat)
    # get the number of students at this school who are above this student's rank
    num_above = np.sum(school_pref_mat[which_school, competitors] < this_student_rank)
    # compare this student's rank to this school's capacity
    safe = num_above < school_capacities[which_school]
    return safe


def generate_cardinal_utility(n_students, n_schools):
    ###################################
    #  preference order set up
    ###################################

    # Each school has distinct mean utility
    # This is to simulate popular and (un)popular schools
    school_mean_u = np.random.randint(low=1, high=n_schools, size=n_schools)

    # similarly, each student has distinct mean utility
    # This is to simulate high-performing and low-performing students
    student_mean_u = np.random.randn(n_students)

    # school preference orders
    school_cardinal_u = np.zeros((n_schools, n_students))
    for i in range(n_schools):
        school_cardinal_u[i, :] = student_mean_u + 2 * np.random.randn(n_students)

    # student preference orders
    student_cardinal_u = np.zeros((n_students, n_schools))
    for i in range(n_students):
        student_cardinal_u[i, :] = school_mean_u + 2 * np.random.randn(n_schools)

    return student_cardinal_u, school_cardinal_u


def generate_school_capacities(n_schools, n_students, school_capacity):
    school_capacities = np.random.randint(
        low=school_capacity[0], high=school_capacity[1], size=n_schools
    )
    while school_capacities.sum() < n_students:
        school_capacities = np.random.randint(
            low=school_capacity[0], high=school_capacity[1], size=n_schools
        )
    return school_capacities


def generate_pref_ordering(student_cardinal_u, school_cardinal_u):

    # school preference orders
    school_pref = np.zeros(school_cardinal_u.shape)
    for i in range(len(school_pref)):
        school_pref[i, :] = np.argsort(school_cardinal_u[i, :])
    school_pref = school_pref.astype(int)

    # student preference orders
    student_pref = np.zeros(student_cardinal_u.shape)
    for i in range(len(student_pref)):
        student_pref[i, :] = np.argsort(student_cardinal_u[i, :])
    student_pref = student_pref.astype(int)

    return student_pref, school_pref


def misreport(
    true_student_pref, soph_students, strat_top, school_pref, school_capacities
):
    reported_student_pref = true_student_pref.copy()

    # Strategies:
    # look at your top [strat_top] choices, and pick the most likely one to report and first chocie
    for student in soph_students:
        safe_list = []
        for i in range(strat_top):
            which_school = reported_student_pref[student, i]
            safe = safe_if_first_choice(
                which_school,
                student,
                school_pref,
                true_student_pref,
                school_capacities,
            )
            safe_list.append(safe)
        safe_list = np.array(safe_list)

        if np.array(safe_list == False).all():
            continue
        else:
            safe_schools = np.where(safe_list == True)[0]
            reach_schools = np.where(safe_list == False)[0]
            old_pref = true_student_pref[student, :].copy()
            new_pref = np.r_[
                old_pref[safe_schools], old_pref[reach_schools], old_pref[strat_top:]
            ]

        # update preference in student_pref
        reported_student_pref[student, :] = new_pref
        reported_student_pref = reported_student_pref.astype(int)

    return reported_student_pref


def run_simulation(n_students, n_schools, school_capacity_range, n_soph):
    """
    verbose = 0: no output
            = 1: print output
            = 2: print round information and output
    """
    n_sinc = n_students - n_soph
    strat_top = 5

    ####################################
    # Set up
    ####################################

    # randomly generate school capacities
    school_capacities = generate_school_capacities(
        n_schools, n_students, school_capacity_range
    )

    # generate student capabilities (always 1)
    student_capacities = np.ones(n_students).astype(int)

    # randomly generate cardinal utility
    student_cardinal_u, school_cardinal_u = generate_cardinal_utility(
        n_students, n_schools
    )

    # generate preference ordering based on cardinal utility
    true_student_pref, school_pref = generate_pref_ordering(
        student_cardinal_u, school_cardinal_u
    )

    # randomly generate sophisticated students
    if n_soph > 0:
        soph_students = random.choices(range(n_students), k=n_soph)

        # generate reported preference ordering
        reported_student_pref = misreport(
            true_student_pref, soph_students, strat_top, school_pref, school_capacities
        )
    else:
        reported_student_pref = true_student_pref.copy()

    ####################################
    # Run simulation
    ####################################

    # Deferred Acceptance
    DA = DeferredAcceptance(
        reported_student_pref, school_pref, student_capacities, school_capacities
    )
    DA_matching, _ = DA.run(verbose=False)

    BM = BostonMechanism(
        reported_student_pref, school_pref, student_capacities, school_capacities
    )
    BM_matching, _ = BM.run(verbose=False)

    ####################################
    # Summarize results
    ####################################
    DA_results, BM_results = {}, {}
    if n_soph == 0:
        soph_students = []
    DA_results["ranking_ave"] = summarzie_ranking(
        DA_matching, true_student_pref, soph_students, np.mean
    )
    BM_results["ranking_ave"] = summarzie_ranking(
        BM_matching, true_student_pref, soph_students, np.mean
    )

    DA_results["ranking_med"] = summarzie_ranking(
        DA_matching, true_student_pref, soph_students, np.median
    )
    BM_results["ranking_med"] = summarzie_ranking(
        BM_matching, true_student_pref, soph_students, np.median
    )

    DA_results["ranking_std"] = summarzie_ranking(
        DA_matching, true_student_pref, soph_students, np.std
    )
    BM_results["ranking_std"] = summarzie_ranking(
        BM_matching, true_student_pref, soph_students, np.std
    )

    return DA_results, BM_results


def summarzie_ranking(matching, true_student_pref, soph_students, func):

    n_students = len(matching)

    # compute average ranking achieved by students in DA
    pref_achieved = np.zeros(n_students) - np.inf
    for student in matching.keys():
        match = matching[student][0][1]
        pref_achieved[student] = np.where(true_student_pref[student] == match)[0][0]
    ave_ranking = func(pref_achieved)

    # compute average ranking achieved by sophisticated/sincere students in DA
    if len(soph_students) > 0 and len(soph_students) < n_students:
        soph_ave_ranking = func(pref_achieved[soph_students])
        sinc_ave_ranking = func(
            pref_achieved[[i for i in range(n_students) if i not in soph_students]]
        )

        return [
            ave_ranking,
            soph_ave_ranking,
            sinc_ave_ranking,
        ]
    else:
        return [ave_ranking]

