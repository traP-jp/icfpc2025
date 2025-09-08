**(This is a translation by ChatGPT.)**


# primus–quintus

We generate route plans at random, trying to keep the usage counts of doors 0–5 as even as possible.
From each expedition’s observations, we *commit* to which underlying room produced that observation, then—per room—compute a frequency distribution over where each door seems to lead. We adopt the modal destination(s) from those distributions to assemble a graph (note this graph may still be directed in places, and some doors may have undetermined destinations). We score how close this graph is to a valid solution and run simulated annealing to minimize that score.

A map is *valid* iff (i) every door’s destination is uniquely determined and (ii) the graph can be treated as undirected.

* The uniqueness condition is equivalent to, for the candidate-destination frequency distribution $d$ of any room’s door,

  $$
  \sum d - \max(d) \;=\; 0.
  $$

* The “undirectable” condition is equivalent to, letting $e_{i,j}$ be the number of directed edges $i \to j$, requiring for every room $u$,

  $$
  \max\!\Bigl(\;\sum_{v \in V}\max\bigl(e_{u,v},\,e_{v,u}\bigr)\;-\;6,\;0\Bigr)\;=\;0.
  $$

  We minimize the sum of these two quantities as our objective.

A single annealing move reassigns the room hypothesized for one observation to a different room. With delta updates, we can evaluate each move in $O(\texttt{SIZE})$.

Because the counts of label values $0,1,2,3$ are fixed by the problem (they’re 2-bit room labels), we never propose moves that would change a room labeled “0” into one labeled “1,” etc. Prioritizing moves whose current frequency distributions have many zeros helps convergence; with $\texttt{SIZE}=30$, we typically find a solution in about 0.1 seconds.

# aleph–he

These instances are built by **vertex doubling**: start from a graph half the size, then duplicate each vertex; connections between the two copies are randomized. On the first *explore*, we recover the pre-doubling graph using the same approach as in *primus–quintus*.

After that, we can determine how doubled edges are wired by rewriting the label on **only one** copy of a vertex and then traversing every edge of the *pre-doubling* graph. A plan built from a TSP pass plus a Chinese Postman (route inspection) walk visits all edges within the plan-length bound; we solved the odd-degree matching for the postman tour by simulated annealing.

# vau–iod

Here the construction is **tripling** rather than doubling. First consider the easier case where the pre-tripling graph has no self-loops.

As before, the first *explore* recovers the pre-tripling graph. Because the hidden permutation $P$ over the three copies satisfies $P_u \ne P_v$ for all distinct $u,v$, it suffices to find one path $p$ that forms a cycle *before* tripling but *not* after tripling. Then we can identify the wiring with the following sequence (all “reverse” steps simply retrace earlier motion):

1. Rewrite a room label (move **a**).
2. Travel to the start of $p$ (move **b**).
3. Traverse $p$.
4. Reverse **b**.
5. While rewriting room labels, reverse **a**.
6. Visit all edges (move **c**).
7. Travel again to the start of $p$ (move **d**).
8. Traverse $p$.
9. Reverse **d**.
10. Reverse **c**.

The plan-length budget is generous enough for this.

To get such a path $p$ without extra explores, pick a cycle from the pre-tripling graph and rely on chance. You need to visit a cycle twice. If you use **different** cycles for the two passes, the success probability is $4/9$; if you use the **same** cycle both times, it’s $5/9$.

Caveat: with self-loops, the simple “reverse playback” may fail, so the sequence above needs adjustments. We ran out of time to implement those refinements, submitted as-is—and it worked.
