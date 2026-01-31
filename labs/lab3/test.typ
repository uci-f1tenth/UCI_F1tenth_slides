#import "@preview/diatypst:0.9.0": *
#show: slides.with(
  title: "Graph SLAM",
  subtitle: "slam_toolbox",
  date: datetime.today().display(),
  authors: "Alistair Keiller",
  title-color: rgb("#1e1e1e"),
  theme: "full",
)
#let argmin = $op("argmin", limits: #true)$
= Pose graph SLAM
== Credit
Wolfram Burgard, Giorgio Grisetti, and Cyrill Stachniss: http://ais.informatik.uni-freiburg.de/teaching/ws11/robotics2/pdfs/ls-slam-tutorial.pdf
== Odometry
- Constraints connect the poses of the robot while it is moving using odometry
- Constraints are inherently uncertain

$ #image("foot.png", height: 70%) $
== Lidar
- Observing previously seen areas generates new constraints
$ #image("lil_guy.png", height: 70%) $

== Idea of Pose Graph-based SLAM
- *Graph:* represents the problem
- *Node:* corresponds to an estimated pose in the robot at a given time
- *Edge:* an approximate spatial constraints between two nodes.
- *Graph-based SLAM:* Build the graph and find a node configuration that minimizes the error introduced by the constraints

== Idea of Pose Graph-based SLAM pt 2
- The nodes represent the state vector
- Given a state, we can compute what we expect to perceive.
- We have real observations that relate nodes with each other
- *Goal:* Find a configuration of the nodes so that the real and predicted observations are as similar as possible.

== Graphical Explanation
$ #image("graphical.png", height: 70%) $

== Least Squares Approach
- Least squares error minimization
$ x^*=argmin_x sum_(i j)e_(i j)^T Omega_(i j) e_(i j) $
- Error function $e_(i j)$ for an observation
$ e_(i j)=(x_j -x_i)-z_(i j) $

== Error function
- The error $e_i$ is typically the difference between the actual and predicted measuremnt
$ e_i (bold(x)) = bold(z)_i-bold(f)_i (bold(x)) $
- We assume that the measurement error has zero mean and is normally distribtued
- Gaussian error with information matrix $Omega_i$
- The squared error of a measurement depends only on the state and is a scalar
$ e_i (bold(x))=e_i (bold(x))^T Omega_i e_i (bold(x)) $

== Goal: Find the minimum
- Find the state $x^*$ that minimizes the error given all the measurements
$ x^*=argmin_x sum_i e^T_i (x) Omega_i e_i (x) $
- A general solution is to derive the global error function and find its nulls
- In general complex and no closed form solution

= Least Squares for SLAM
== Solving with least squares
$ #image("closure.png") $

== The pose graph
- It consistants of $n$ nodes $x=x_(1:n)$
- Each $x_i$ is a robot pose (at time $t_i$)
- We create an edge between nodes $x_i$ and $x_j$ if and only if...

== Create an Edge If... (1)
- ...The robot moves from $x_i$ to $x_(i+1)$
- Edge corresponds to odometry data
$ #image("odom_edge.png") $

== Create an Edge If... (2)
- The robot observes the same part of the environment in both $x_i$ and $x_j$.
$ #image("closure_edge.png", height: 50%) $
$ #image("scan_matching_edge.png", height: 30%) $

== Pose Graph
$ #image("pose_graph.png") $



== Minimization
- Open problem. You can use gauss-newton, or a minimization library (slam_toolbox uses ceres)
