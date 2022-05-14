def align(nbest, nBest, placeholder="-"):
    """
    input :
    [
        [a,b,c,d]  # hyp1
        [a,b,d,e]  # hyp2
        [a,b,c,d,e]# hyp3
    ]

    output:
    [
        [
            [a,a], [b,b], [c, -], [d,d], [-, e]
        ],
        [
            [a,a], [b,b]. [c,c], [d,d], [-, e]
        ]
    ]
    """
    pair = []
    for candidate in nbest[1:nBest]:
        pair.append([nbest[0], candidate])

    # First, align them pair by pair -- using minimum edit distance
    first_align = []
    for a in pair:
        cost_matrix = [[0 for i in range(len(a[1]) + 1)] for j in range(len(a[0]) + 1)]
        for j in range(len(a[1]) + 1):
            cost_matrix[0][j] = j
        for i in range(len(a[0]) + 1):
            cost_matrix[i][0] = i

        for i in range(0, len(a[0])):
            for j in range(0, len(a[1])):
                if a[0][i] == a[1][j]:
                    cost_matrix[i + 1][j + 1] = min(
                        cost_matrix[i][j],
                        cost_matrix[i + 1][j] + 1,
                        cost_matrix[i][j + 1] + 1,
                    )
                else:
                    cost_matrix[i + 1][j + 1] = min(
                        cost_matrix[i][j] + 2,
                        cost_matrix[i + 1][j] + 1,
                        cost_matrix[i][j + 1] + 1,
                    )

        l1 = len(a[0]) - 1
        l2 = len(a[1]) - 1

        align_result = []
        while l1 >= 0 and l2 >= 0:
            if a[0][l1] == a[1][l2]:
                cost = 0
            else:
                cost = 2

            r = cost_matrix[l1 + 1][l2 + 1]
            diag = cost_matrix[l1][l2]
            left = cost_matrix[l1 + 1][l2]
            down = cost_matrix[l1][l2 + 1]

            if r == diag + cost:
                align_result = [[a[0][l1], a[1][l2]]] + align_result
                l1 -= 1
                l2 -= 1

            else:
                if r == left + 1:
                    align_result = [[placeholder, a[1][l2]]] + align_result
                    l2 -= 1
                else:
                    align_result = [[a[0][l1], placeholder]] + align_result
                    l1 -= 1
        if l1 < 0:
            while l2 >= 0:
                align_result = [[placeholder, a[1][l2]]] + align_result
                l2 -= 1
        elif l2 < 0:
            while l1 >= 0:
                align_result = [[a[0][l1], placeholder]] + align_result
                l1 -= 1
        first_align.append(align_result)

    return first_align


def alignNbest(nbestAlign, placeholder="-"):
    """
    input:
    [
        [
            [a,a], [b,b], [c,c], [d,-],[-,e]
        ],
        [
            [a,-], [b,b], [c,c], [d,d]
        ]
    ]

    output:
    [
        [a,a,-],[b,b,b],[c,c,c],[d,-,d],[-,e,-]
    ]
    """

    alignResult = nbestAlign[0]
    for a in nbestAlign[1:]:
        ali = [alignResult, a]
        l1 = 0
        l2 = 0
        align = []
        while l1 < len(ali[0]) and l2 < len(ali[1]):
            if ali[0][l1][0] == ali[1][l2][0]:
                align.append(ali[0][l1] + ali[1][l2][1:])
                l1 += 1
                l2 += 1
            else:
                if ali[0][l1][0] == placeholder:
                    align.append(
                        [placeholder]
                        + ali[0][l1][1:]
                        + [placeholder for _ in range(len(ali[1][l2]) - 1)]
                    )
                    l1 += 1
                else:
                    align.append(
                        [placeholder]
                        + [placeholder for _ in range(len(ali[0][l1]) - 1)]
                        + ali[1][l2][1:]
                    )
                    l2 += 1

            if l1 == len(ali[0]):
                while l2 < len(ali[1]):
                    align.append(
                        [placeholder]
                        + [placeholder for _ in range(len(ali[0][l1 - 1]) - 1)]
                        + ali[1][l2][1:]
                    )
                    l2 += 1
            elif l2 == len(ali[1]):
                while l1 < len(ali[0]):
                    align.append(
                        [placeholder]
                        + ali[0][l1][1:]
                        + [placeholder for _ in range(len(ali[1][l2 - 1]) - 1)]
                    )
                    l1 += 1

        alignResult = align

    return alignResult
