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
== The pose graph
- It consistants of $n$ nodes $x=x_(1:n)$
- Each $x_i$ is a robot pose (at time $t_i$)
- We create an edge between nodes $x_i$ and $x_j$ if and only if...

== Create an Edge If... (1)
- ...The robot moves from $x_i$ to $x_(i+1)$
- Edge corresponds to odometry data
$ #image("odom_edge.png") $

== Odometry
- Constraints connect the poses of the robot while it is moving using odometry
- Constraints are inherently uncertain

$ #image("foot.png", height: 70%) $

== Create an Edge If... (2)
- The robot observes the same part of the environment in both $x_i$ and $x_j$.
$ #image("closure_edge.png", height: 50%) $
$ #image("scan_matching_edge.png", height: 30%) $

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

- We have real observations that relate nodes to each other

- *Goal:* Find a configuration of the nodes so that the real and predicted observations are as similar as possible.

== Graphical Explanation
$ #image("graphical.png", height: 70%) $

== Solving with least squares
$ #image("closure.png") $

== Least Squares Approach
- Least squares error minimization
$ x^*=argmin_x sum_(i j)e_(i j)^T Omega_(i j) e_(i j) $
- Error function $e_(i j)$ for an observation
$ e_(i j)=(x_j -x_i)-z_(i j) $

== Pose Graph
$ #image("pose_graph.png") $

== Minimization

- Open problem. You can use Gauss-Newton if you were doing this from scratch, but ideally, use a minimization library (slam_toolbox uses Ceres).
$ #image("optimization.png", height: 80%) $

== Removing outliers
$ #image("outliers.png") $

= Scan matching
== What is scan matching
- Scan matching creates the lidar edges between $x_i$ and $x_j$ by finding lidar observations of the same object.
- There are many different ways to do it:
  - Iterative closest point (ICP)
  - Scan-to-scan
  - Scan-to-map
  - Map-to-map
  - Feature-based
  - RANSAC for outlier rejection
  - Correlative matching
== Iterative Closest Point
- ICP is a way to match two point clouds.
$ #image("ICP.png") $
== Data Association
- For each point $a_i$ on point cloud $a$, find the closest point $b_j$.
$ #image("data_association.png", height: 80%) $
== Transformation
- Find a transformation to align the two point clouds

- First, align the center of mass of both point clouds, and then rotate them using SVD.

$ #image("transformation.png", height: 75%) $
== Iterate
- Keep going until they converge!
$ #image("converge.png", height: 80%) $
= All Together
== Loop Closure
$ #image("GBS.gif", height: 80%) $
== Slam Toolbox
$ #image("slam_toolbox.gif", height: 80%) $
