import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from BostonMech import BostonMechanism
from DA import DeferredAcceptance
from ChineseParallel import ChineseParallel


def get_rank_at_school(which_school, which_student, school_pref_mat):
    return np.where(school_pref_mat[which_school] == which_student)[0][0]


def safe_if_first_choice(
    which_school, which_student, school_pref_mat, student_pref_mat, school_capacities
):
    competitors = np.where(student_pref_mat[:, 0] == which_school)
    # get this student's rank at this school
    this_student_rank = get_rank_at_school(which_school, which_student, school_pref_mat)
    # print(this_student_rank)
    # get the number of students at this school who are above this student's rank
    num_above = (
        np.sum(school_pref_mat[which_school, competitors] < this_student_rank)
        + 5 * np.random.randn()
    )
    # print(num_above)
    # compare this student's rank to this school's capacity
    safe = num_above < school_capacities[which_school]
    return safe


def gen_siblings(n_students, n_schools):
    has_sibling = np.where(np.random.uniform(size=n_students) < 0.3, 1, 0)
    sibling_school = np.random.choice(range(n_schools), size=n_students)
    sibling_school = np.where(has_sibling == 1, sibling_school, -99)
    return sibling_school


def gen_sibling_priority(sibling_school, n_schools):
    """returns np.array of size (n_schools, n_students)"""
    n_students = len(sibling_school)
    sibling_priority = np.zeros((n_schools, n_students)) - np.inf
    for school in range(n_schools):
        sibling_priority[school, :] = sibling_school == school
    return sibling_priority


def gen_student_district(n_students):
    """
    Return an np.array of len(n_students) with values of 0 or 1
    1 = in East Boston 
    0 = not in East Boston
    """
    district = np.random.binomial(1, 1 / 10, size=n_students)
    return district


def gen_distance_priority(school_dist, student_dist):
    """
    Return an np.array of size (n_schools, n_students) with values of 0 or 1
    1 = student and school in the same side of Bostons
    0 = otherwise
    """
    n_students = len(student_dist)
    n_schools = len(school_dist)
    dist_prio = np.zeros((n_schools, n_students)) - np.inf
    for sch in range(n_schools):
        dist_prio[sch, :] = school_dist[sch] == student_dist
    return dist_prio


def gen_priority(sibling_info, dist_info):
    # treat sibling prioity as lexiographically higher than district priority
    priority_score = 2 * sibling_info + dist_info
    return priority_score


def gen_school_card_u(priority_score):
    # add a noise from uniform(0,1) to break ties
    return priority_score + np.random.uniform(size=priority_score.shape)


def load_school_info():
    # use Boston Open Enrollment data to generate school quality
    # key = school name
    # value = school quality (lower is better)
    quality_dict = {
        "Another Course to College": 2,
        "Boston International Newcomers Academy": 4,
        "Brighton High School": 4,
        "Jeremiah E. Burke High School": 2,
        "Charlestown High School": 3,
        "Community Academy of Science & Health (CASH)": 4,
        "Dearborn STEM 6-12 Early College Academy": 4,
        "East Boston High School": 3,
        "The English High School": 4,
        "Excel High School": 2,
        "Henderson K-12 Inclusion School": 2,
        "Lyon Pilot High School": 4,
        "Madison Park Technical Vocational High School": 4,
        "Margarita Muñiz Academy": 3,
        "Quincy Upper School": 2,
        "Snowden International School at Copley": 4,
        "TechBoston Academy": 3,
    }

    enrollment_dict = {
        "Another Course to College": 244 / 4,
        "Boston International Newcomers Academy": 345 / 4,
        "Brighton High School": 408 / 4,
        "Jeremiah E. Burke High School": 402 / 4,
        "Charlestown High School": 793 / 6,
        "Community Academy of Science & Health (CASH)": 342 / 4,
        "Dearborn STEM 6-12 Early College Academy": 581 / 6,
        "East Boston High School": 1051 / 6,
        "The English High School": 527 / 4,
        "Excel High School": 474 / 4,
        "Henderson K-12 Inclusion School": 932 / 12,
        "Lyon Pilot High School": 142 / 4,
        "Madison Park Technical Vocational High School": 1066 / 4,
        "Margarita Muñiz Academy": 315 / 4,
        "Quincy Upper School": 535 / 7,
        "Snowden International School at Copley": 487 / 4,
        "TechBoston Academy": 913 / 7,
    }

    geo_dict = {
        "Another Course to College": 0,
        "Boston International Newcomers Academy": 0,
        "Brighton High School": 0,
        "Jeremiah E. Burke High School": 0,
        "Charlestown High School": 0,
        "Community Academy of Science & Health (CASH)": 0,
        "Dearborn STEM 6-12 Early College Academy": 0,
        "East Boston High School": 1,
        "The English High School": 0,
        "Excel High School": 0,
        "Henderson K-12 Inclusion School": 0,
        "Lyon Pilot High School": 0,
        "Madison Park Technical Vocational High School": 0,
        "Margarita Muñiz Academy": 0,
        "Quincy Upper School": 0,
        "Snowden International School at Copley": 0,
        "TechBoston Academy": 0,
    }

    school_names = list(quality_dict.keys())
    school_quality = list(quality_dict.values())
    school_capacities = [math.ceil(x) for x in list(enrollment_dict.values())]
    school_east = list(geo_dict.values())
    return (
        np.array(school_names),
        10 - np.array(school_quality),  # reverse quality so that bigger is better
        np.array(school_capacities),
        np.array(school_east),
    )


def gen_student_card_u(n_students, school_quality):
    umat = np.zeros((n_students, len(school_quality))) - np.inf
    for stu in range(n_students):
        umat[stu, :] = school_quality

    # add a noise from standard normal as student's utility from the school
    umat = umat + 0.5 * np.random.randn(*umat.shape)
    return umat


def gen_order_pref(cardinal_u):

    # school preference orders
    order_pref = np.zeros(cardinal_u.shape)
    for i in range(len(order_pref)):
        order_pref[i, :] = np.argsort(-cardinal_u[i, :])
    order_pref = order_pref.astype(int)

    return order_pref


def misreport1(true_student_pref, soph_students, school_pref, school_capacities):
    strat_top = 3
    reported_student_pref = true_student_pref.copy()

    # Strategies:
    # One school choice strategy is to find a school you like that is undersubscribed and put it as a top choice
    for student in soph_students:
        safe_school = None
        for i in range(strat_top):
            which_school = reported_student_pref[student, i]
            safe = safe_if_first_choice(
                which_school,
                student,
                school_pref,
                true_student_pref,
                school_capacities,
            )
            if safe:
                safe_school = which_school
                break

        if safe_school is None:  # no safe school for this student
            continue
        elif safe_school == true_student_pref[student, 0]:  # first choice is safe
            continue
        else:  # this student has a safe school
            old_pref = true_student_pref[student, :].copy()
            new_pref = np.array(
                [safe_school] + list(old_pref[0:i]) + list(old_pref[i + 1 :])
            )

        # update preference in student_pref
        reported_student_pref[student, :] = new_pref
        reported_student_pref = reported_student_pref.astype(int)

    return reported_student_pref


def misreport2(true_student_pref, soph_students, school_pref, school_capacities):

    # strategy:
    # find a school that you like that is popular and put it as a first choice
    # and find a school that is less popular for a “safe” second choice.

    strat_top = 5
    reported_student_pref = true_student_pref.copy()

    # Strategies:
    # One school choice strategy is to find a school you like that is undersubscribed and put it as a top choice
    for student in soph_students:
        safe_school = None
        for i in range(1, strat_top):
            which_school = reported_student_pref[student, i]
            safe = safe_if_first_choice(
                which_school,
                student,
                school_pref,
                true_student_pref,
                school_capacities,
            )
            if safe:
                safe_school = which_school
                break

        if safe_school is None:  # no safe school for this student
            continue
        elif (
            safe_school == true_student_pref[student, 1]
        ):  # safe choice is the second choice: no change in reported preference
            continue
        else:  # if safe schoice is >second choice, change reported preference
            new_pref = true_student_pref[student, :].copy()
            new_pref[1], new_pref[i] = safe_school, new_pref[1]

            old_pref = true_student_pref[student, :].copy()
            new_pref = np.array(
                [old_pref[0]]
                + [safe_school]
                + list(old_pref[1:i])
                + list(old_pref[i + 1 :])
            )

        # update preference in student_pref
        reported_student_pref[student, :] = new_pref
        reported_student_pref = reported_student_pref.astype(int)

    return reported_student_pref


def misreport3(true_student_pref, soph_students, school_pref, school_capacities):

    # strategy:
    # Among top 5 schools, pick the top 3 safe school and report them as the first 3 schools

    strat_top = 5
    reported_student_pref = true_student_pref.copy()

    # Strategies:
    # One school choice strategy is to find a school you like that is undersubscribed and put it as a top choice
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
            # as soon as we find 3 safe schools, break
            if len(np.array(safe_list) == True) >= 3:
                break
        safe_list = np.array(safe_list)

        # if no safe schools, no change in reported preference
        if np.array(safe_list == False).all():
            continue
        else:
            safe_schools = np.where(safe_list == True)[0]
            reach_schools = np.where(safe_list == False)[0]
            old_pref = true_student_pref[student, :].copy()
            # misreport the first 3 safe schools as first 3 schools
            new_pref = np.r_[
                old_pref[safe_schools], old_pref[reach_schools], old_pref[strat_top:]
            ]

        # update preference in student_pref
        reported_student_pref[student, :] = new_pref
        reported_student_pref = reported_student_pref.astype(int)

    return reported_student_pref


def run_simulation(
    boston_school_info,
    n_schools,
    n_students,
    student_capacities,
    perc_soph,
    soph1_idx,
    soph2_idx,
):

    # load Boston school info
    school_names = boston_school_info[0]
    school_quality = boston_school_info[1]
    school_capacities = boston_school_info[2]
    school_east = boston_school_info[3]

    # get slibling info
    sibling_school = gen_siblings(n_students, n_schools)
    sibling_prio = gen_sibling_priority(sibling_school, n_schools)
    # get distance info
    student_dist = gen_student_district(n_students)
    dist_info = gen_distance_priority(school_east, student_dist)
    # compute priority
    priority = gen_priority(sibling_prio, dist_info)
    # compute school cardinal utility
    school_cardinal = gen_school_card_u(priority)
    school_ordinal = gen_order_pref(school_cardinal)

    # generate student preference over schools
    student_cardinal = gen_student_card_u(n_students, school_quality)
    student_ordinal = gen_order_pref(student_cardinal)

    # gather sophisticated students index
    soph_idx = np.array(list(soph1_idx) + list(soph2_idx))
    n_soph = len(soph_idx)

    # generate reported preference ordering
    reported_student_pref_BM = misreport1(
        student_ordinal, soph1_idx, school_ordinal, school_capacities
    )
    reported_student_pref_BM = misreport2(
        reported_student_pref_BM, soph2_idx, school_ordinal, school_capacities
    )
    reported_student_pref_CP = misreport2(
        student_ordinal, soph_idx, school_ordinal, school_capacities
    )

    # Deferred Acceptance
    DA = DeferredAcceptance(
        student_ordinal, school_ordinal, student_capacities, school_capacities
    )
    DA_matching, DA_matching_school = DA.run(verbose=False)

    BM = BostonMechanism(
        reported_student_pref_BM, school_ordinal, student_capacities, school_capacities
    )
    BM_matching, BM_matching_school = BM.run(verbose=False)

    CP = ChineseParallel(
        reported_student_pref_BM, school_ordinal, student_capacities, school_capacities
    )
    CP_matching, CP_matching_school = CP.run(breaks=[3, 6])

    ####################################
    # Summarize results
    ####################################
    DA_results, BM_results, CP_results = {}, {}, {}

    if n_soph == 0:
        soph_idx = []

    DA_results["ranking_ave"] = summ_ranking(
        DA_matching, student_ordinal, soph_idx, np.mean
    )
    BM_results["ranking_ave"] = summ_ranking(
        BM_matching, student_ordinal, soph_idx, np.mean
    )
    CP_results["ranking_ave"] = summ_ranking(
        CP_matching, student_ordinal, soph_idx, np.mean
    )

    DA_results["ranking_med"] = summ_ranking(
        DA_matching, student_ordinal, soph_idx, np.median
    )
    BM_results["ranking_med"] = summ_ranking(
        BM_matching, student_ordinal, soph_idx, np.median
    )
    CP_results["ranking_med"] = summ_ranking(
        CP_matching, student_ordinal, soph_idx, np.median
    )

    DA_results["ranking_std"] = summ_ranking(
        DA_matching, student_ordinal, soph_idx, np.std
    )
    BM_results["ranking_std"] = summ_ranking(
        BM_matching, student_ordinal, soph_idx, np.std
    )
    CP_results["ranking_std"] = summ_ranking(
        CP_matching, student_ordinal, soph_idx, np.std
    )

    DA_results["frac_choice_achieved"] = summ_fraction_choice_ahieved(
        DA_matching, student_ordinal, soph_idx
    )
    BM_results["frac_choice_achieved"] = summ_fraction_choice_ahieved(
        BM_matching, student_ordinal, soph_idx
    )
    CP_results["frac_choice_achieved"] = summ_fraction_choice_ahieved(
        CP_matching, student_ordinal, soph_idx
    )

    DA_results["u_ave"] = sum_utility(DA_matching, student_cardinal, soph_idx, np.mean)
    BM_results["u_ave"] = sum_utility(BM_matching, student_cardinal, soph_idx, np.mean)
    CP_results["u_ave"] = sum_utility(CP_matching, student_cardinal, soph_idx, np.mean)

    DA_results["u_med"] = sum_utility(
        DA_matching, student_cardinal, soph_idx, np.median
    )
    BM_results["u_med"] = sum_utility(
        BM_matching, student_cardinal, soph_idx, np.median
    )
    CP_results["u_med"] = sum_utility(
        CP_matching, student_cardinal, soph_idx, np.median
    )

    DA_results["u_std"] = sum_utility(DA_matching, student_cardinal, soph_idx, np.std)
    BM_results["u_std"] = sum_utility(BM_matching, student_cardinal, soph_idx, np.std)
    CP_results["u_std"] = sum_utility(CP_matching, student_cardinal, soph_idx, np.std)

    rank_improv = {}
    rank_improv["BM2DA"] = rank_improvement(DA_matching, BM_matching, student_ordinal)
    rank_improv["BM2CP"] = rank_improvement(CP_matching, BM_matching, student_ordinal)

    u_improv = u_improvement(DA_matching, BM_matching, student_cardinal)

    n_blocks = {}
    n_blocks["DA"] = count_blocking_pars(
        DA_matching, DA_matching_school, student_ordinal, school_cardinal
    )
    n_blocks["BM"] = count_blocking_pars(
        BM_matching, BM_matching_school, student_ordinal, school_cardinal
    )
    n_blocks["CP"] = count_blocking_pars(
        CP_matching, CP_matching_school, student_ordinal, school_cardinal
    )

    return DA_results, BM_results, CP_results, rank_improv, u_improv, soph_idx, n_blocks


def count_blocking_pars(student_match, school_match, student_ordinal, school_cardinal):

    n_students = len(student_ordinal)
    n_schools = len(school_cardinal)

    blocks = []
    for i in range(n_students):
        for s in range(n_schools):
            # check if i, s is blocking pair
            if student_match[i][0][1] == s:
                continue
            match = student_match[i][0][1]

            # does i prefer match to s?
            pref = student_ordinal[i]
            if (pref == match).argmax() < (pref == s).argmax():
                continue

            # does s prefer match to i?
            s_matches = school_match[s]
            for j in s_matches:
                # check if s likes j more than i
                if school_cardinal[s, j[1]] < school_cardinal[s, i]:
                    blocks.append((i, s))
                    break
    return len(blocks)


def rank_improvement(matching_after, matching_before, true_student_pref):

    rank_improvement = np.zeros(len(matching_after.keys())) - np.inf
    for student in matching_after.keys():
        match_after = matching_after[student][0][1]
        match_before = matching_before[student][0][1]
        after_true_pref = true_pref_ranking(true_student_pref, student, match_after)

        before_true_pref = true_pref_ranking(true_student_pref, student, match_before)

        rank_improvement[student] = after_true_pref - before_true_pref

        # print(DA_true_pref, BM_true_pref)
    return rank_improvement


def u_improvement(DA_matching, BM_matching, student_cardinal):

    u_improvement = np.zeros(len(DA_matching.keys())) - np.inf
    for student in DA_matching.keys():
        match_DA = DA_matching[student][0][1]
        match_BM = BM_matching[student][0][1]
        u_improvement[student] = (
            student_cardinal[student, match_DA] - student_cardinal[student, match_BM]
        )
    return u_improvement


def true_pref_ranking(true_student_pref, student, match):
    # student and match are both indices
    true_order = np.where(true_student_pref[student] == match)[0][0]
    # print(true_student_pref[student], match, true_order)
    return true_order


def summ_ranking(matching, true_student_pref, soph_students, func):

    n_students = len(matching)
    # compute average ranking achieved by students in DA
    pref_achieved = np.zeros(n_students) - np.inf
    for student in matching.keys():
        match = matching[student][0][1]
        pref_achieved[student] = true_pref_ranking(true_student_pref, student, match)
    ave_ranking = func(pref_achieved)

    # compute average ranking achieved by sophisticated/sincere students in DA
    if len(soph_students) > 0 and len(soph_students) < n_students:
        soph_ave_ranking = func(pref_achieved[soph_students])
        sinc_ave_ranking = func(
            pref_achieved[[i for i in range(n_students) if i not in soph_students]]
        )

        return {
            "all": ave_ranking,
            "sophisticated": soph_ave_ranking,
            "sincere": sinc_ave_ranking,
        }
    else:
        return {"all": ave_ranking}


def sum_utility(matching, student_card_u, soph_students, func):

    n_students = len(matching)

    # compute average ranking achieved by students in DA
    u_achieved = np.zeros(n_students) - np.inf
    for student in matching.keys():
        match = matching[student][0][1]
        u_achieved[student] = student_card_u[student, match]
    u = func(u_achieved)

    # compute average ranking achieved by sophisticated/sincere students in DA
    if len(soph_students) > 0 and len(soph_students) < n_students:
        soph_u = func(u_achieved[soph_students])
        sinc_u = func(
            u_achieved[[i for i in range(n_students) if i not in soph_students]]
        )

        return {
            "all": u,
            "sophisticated": soph_u,
            "sincere": sinc_u,
        }
    else:
        return {"all": u}


def summ_fraction_choice_ahieved(matching, true_student_pref, soph_students):
    n_schools = len(true_student_pref[0, :])

    choice_freq = np.zeros(n_schools)
    choice_freq_soph = np.zeros(n_schools)
    choice_freq_sinc = np.zeros(n_schools)

    for student in matching.keys():
        match = matching[student][0][1]

        order_achieved = true_pref_ranking(true_student_pref, student, match)
        choice_freq[order_achieved] += 1

        if student in soph_students:
            order_achieved = true_pref_ranking(true_student_pref, student, match)
            choice_freq_soph[order_achieved] += 1
        else:
            order_achieved = true_pref_ranking(true_student_pref, student, match)
            choice_freq_sinc[order_achieved] += 1

    choice_freq = choice_freq / len(matching)
    choice_freq_soph = choice_freq_soph / len(soph_students)
    choice_freq_sinc = choice_freq_sinc / (len(matching) - len(soph_students))

    if len(soph_students) > 0 and len(soph_students) < len(matching):
        return {
            "all": choice_freq,
            "sophisticated": choice_freq_soph,
            "sincere": choice_freq_sinc,
        }
    else:
        return {"all": choice_freq}


def simulation_wrapper(perc_soph, n_iter, seed=42):
    np.random.seed(seed)
    # load Boston school info
    (school_names, school_quality, school_capacities, school_east,) = load_school_info()
    boston_school_info = (school_names, school_quality, school_capacities, school_east)
    n_schools = len(school_names)
    n_students = school_capacities.sum()
    student_capacities = np.ones(n_students).astype(int)

    # generate sophisticated students index
    n_soph = int(perc_soph * n_students)
    soph_idx = np.random.choice(n_students, n_soph, replace=False)
    soph1_idx = np.random.choice(soph_idx, n_soph // 2, replace=False)
    soph2_idx = np.array([idx for idx in soph_idx if idx not in soph1_idx])

    (
        DA_results,
        BM_results,
        CP_results,
        rank_improvs_DA,
        rank_improvs_CP,
        u_improvs,
        n_blocks_DA,
        n_blocks_BM,
        n_blocks_CP,
    ) = ([], [], [], [], [], [], [], [], [])
    for it in tqdm(range(n_iter)):

        (
            DA_result,
            BM_result,
            CP_result,
            rank_improv,
            u_improv,
            soph_idx,
            n_block,
        ) = run_simulation(
            boston_school_info,
            n_schools,
            n_students,
            student_capacities,
            perc_soph,
            soph1_idx,
            soph2_idx,
        )

        DA_results.append(DA_result)
        BM_results.append(BM_result)
        CP_results.append(CP_result)
        rank_improvs_DA.append(rank_improv["BM2DA"])
        rank_improvs_CP.append(rank_improv["BM2CP"])
        u_improvs.append(u_improv)
        n_blocks_DA.append(n_block["DA"])
        n_blocks_BM.append(n_block["BM"])
        n_blocks_CP.append(n_block["CP"])

    DA_results = np.array(DA_results)
    BM_results = np.array(BM_results)
    CP_results = np.array(CP_results)
    rank_improvs_DA = np.array(rank_improvs_DA)
    rank_improvs_CP = np.array(rank_improvs_CP)
    u_improvs = np.array(u_improvs)
    n_blocks_DA = np.array(n_blocks_DA)
    n_blocks_BM = np.array(n_blocks_BM)
    n_blocks_CP = np.array(n_blocks_CP)

    return [
        DA_results,
        BM_results,
        CP_results,
        rank_improvs_DA,
        rank_improvs_CP,
        n_blocks_DA,
        n_blocks_BM,
        n_blocks_CP,
        soph_idx,
    ]


def plot_frac_choice(results, group, n_iter, plot_related):

    canvas = plot_related[0]
    plot_title = plot_related[1]
    plot_y_label = plot_related[2]
    plot_x_label = plot_related[3]
    plot_color = plot_related[4]
    plot_yrange = plot_related[5]

    res_DA = np.zeros((n_iter, len(results[0]["frac_choice_achieved"][group])))

    for i in range(n_iter):
        res_DA[i, :] = results[i]["frac_choice_achieved"][group]

    res_DA = np.mean(res_DA, axis=0)

    x = np.arange(len(res_DA))
    width = 0.7

    canvas.bar(x, res_DA, width, color=plot_color, alpha=0.6)
    canvas.set_xticks(x, x + 1)
    canvas.set(title=plot_title, ylabel=plot_y_label, xlabel=plot_x_label)
    canvas.set_ylim(*plot_yrange)

    return res_DA


def plot_results(results, n_iter, metric, plot_related, CP=False):
    DA_results, BM_results, CP_results = results

    title = plot_related[0]
    y_label = plot_related[1]
    ticks = plot_related[2]
    legend = plot_related[3]
    canvas = plot_related[4]

    nvals = len(DA_results[0][metric]) - 1
    DA_vals = np.zeros((n_iter, nvals))
    BM_vals = np.zeros((n_iter, nvals))
    CP_vals = np.zeros((n_iter, nvals))

    for it in range(n_iter):
        DA_vals[it, :] = np.array(list(DA_results[it][metric].values())[1:])
        BM_vals[it, :] = np.array(list(BM_results[it][metric].values())[1:])
        CP_vals[it, :] = np.array(list(CP_results[it][metric].values())[1:])

    mean_DA_vals = DA_vals.mean(axis=0)
    mean_BM_vals = BM_vals.mean(axis=0)
    mean_CP_vals = CP_vals.mean(axis=0)
    std_DA_val = DA_vals.std(axis=0)
    std_BM_val = BM_vals.std(axis=0)
    std_CP_val = CP_vals.std(axis=0)
    if CP:
        x = np.array(range(3 * nvals))
        mean_DA_vals_plot = np.array(list(mean_DA_vals) + [0, 0, 0, 0])
        mean_BM_vals_plot = np.array([0, 0] + list(mean_BM_vals) + [0, 0])
        mean_CP_vals_plot = np.array([0, 0, 0, 0] + list(mean_CP_vals))
        std_DA_val_plot = np.array(list(std_DA_val) + [0, 0, 0, 0])
        std_BM_val_plot = np.array([0, 0] + list(std_BM_val) + [0, 0])
        std_CP_val_plot = np.array([0, 0, 0, 0] + list(std_CP_val))

        canvas.bar(
            x,
            mean_DA_vals_plot,
            width=0.6,
            color="darkblue",
            alpha=0.6,
            yerr=1.96 * std_DA_val_plot,
            capsize=8,
            label="Deferred Acceptance",
        )
        canvas.bar(
            x,
            mean_BM_vals_plot,
            width=0.6,
            color="maroon",
            alpha=0.6,
            yerr=1.96 * std_BM_val_plot,
            capsize=8,
            label="Boston Mechanism",
        )
        canvas.bar(
            x,
            mean_CP_vals_plot,
            width=0.6,
            color="chocolate",
            alpha=0.6,
            yerr=1.96 * std_CP_val_plot,
            capsize=8,
            label="Chinese Parallel",
        )
        canvas.set_xticks(range(3 * nvals), ticks)
        if legend:
            canvas.legend(bbox_to_anchor=(0.6, -0.18), ncol=3)
        canvas.set(title=title, ylabel=y_label, xlabel="student types")
    else:
        x = np.array(range(2 * nvals))
        mean_DA_vals_plot = np.array(list(mean_DA_vals) + [0, 0])
        mean_BM_vals_plot = np.array([0, 0] + list(mean_BM_vals))
        std_DA_val_plot = np.array(list(std_DA_val) + [0, 0])
        std_BM_val_plot = np.array([0, 0] + list(std_BM_val))

        canvas.bar(
            x,
            mean_DA_vals_plot,
            width=0.6,
            color="darkblue",
            alpha=0.6,
            yerr=1.96 * std_DA_val_plot,
            capsize=8,
            label="Deferred Acceptance",
        )
        canvas.bar(
            x,
            mean_BM_vals_plot,
            width=0.6,
            color="maroon",
            alpha=0.6,
            yerr=1.96 * std_BM_val_plot,
            capsize=8,
            label="Boston Mechanism",
        )
        canvas.set_xticks(range(2 * nvals), ticks)
        if legend:
            canvas.legend(bbox_to_anchor=(0.38, -0.18), ncol=2)
        canvas.set(title=title, ylabel=y_label, xlabel="student types")


def plot_hist(to_plot, plot_related):
    print("(min, max):", (to_plot.min(), to_plot.max()))
    canvas = plot_related[0]
    plot_title = plot_related[1]
    plot_y_label = plot_related[2]
    plot_x_label = plot_related[3]
    plot_label = plot_related[4]
    plot_color = plot_related[5]
    plot_bins = plot_related[6]

    sns.histplot(
        to_plot,
        color=plot_color,
        stat="percent",
        bins=plot_bins,
        element="step",
        alpha=0.2,
        label=plot_label,
    )
    canvas.set(title=plot_title, ylabel=plot_y_label, xlabel=plot_x_label)
    canvas.legend()
