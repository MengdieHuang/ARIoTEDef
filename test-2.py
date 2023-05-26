\section{Architecture of \sys}
\label{sec:overview}

This section presents the design of \sys. We first provide the threat model and the main properties, then describe the architecture of \sys. 

\begin{figure}[ht]
  \centering
  \includegraphics[width=\linewidth]{images/architecture.eps}
  \caption{Architecture of \sys and Infection identification flow.}
  \label{fig:algorithm}
  \vspace{-20pt}
\end{figure}

\subsection{Design Principles} 

\ptitle{Threat model.} 
\sys analyzes network packets exchanged between the network (where it aims to protect) and the Internet. We assume that \sys is not compromised; thus, it does not manipulate the exchanged packets. Also, we assume that an attack is always initiated from the Internet by using remote network access. IoT devices when initially deployed in the network are not compromised; however, they can be compromised later on. 

\ptitle{Main properties.} 
\sys is designed to adhere to the following properties:
\begin{itemize}
    \item \textbf{Network-based:} it works with network packets; thus, it does not impose any computation overhead on IoT devices, and is immediately deployable as it does not require any change on IoT devices.
    \item \textbf{Anomaly-based:} it is able to detect unknown patterns and it is also appropriate for the simple  communication behavior of IoT devices. 
    \item \textbf{Kill chain-based:} it understands multi-step attacks based on a kill chain and deploys classifiers specialized for the steps. 
    \item \textbf{Infection-identifying:} it backtracks past events to identify infection events when it detects known events of later stages. 
    \item \textbf{Self-evolving:} it updates the infection detector with the identified events.
\end{itemize}

\subsection{Overview} 
\sys consists of four main components: \textit{window manager}, \textit{per-step detectors}, \textit{sequence analyzer}, and \textit{detector updater} (see \autoref{fig:algorithm})

\ptitle{Window manager (packets $\rightarrow$ window).}
\sys works on a \textit{flow-based window} where a \textit{flow} is defined as a 5-tuple -- the protocol in use, source/destination IP addresses, and source/destination ports. A window manager collects packets per flow and runs a sliding window based on two parameters -- a \textit{window output period} and a \textit{window length}. On every window output period, the window manager outputs a \textit{window} in the form of a vector that consists of the 84 flow feature values considered in \textsc{CICFlowMeter}~\cite{cicflowmeter}, a network traffic flow analyzer. The flow feature values are evaluated from packets within a window length. For instance, let a window output period be $2$ and a window length be $5$. When a window is output at time $t=5$, the window contains the flow feature values from packets captured between $t=0$ and $t=5$. The next window is output at $t=7$ with the values from packets captured between $t=2$ and $t=7$.

\ptitle{Per-step detectors (window $\rightarrow$ event).}
The main purpose of this component is to map a window to one or more kill chain steps. To this end, \sys has three \textit{per-step detectors} - one for each of the \textit{Reconnaissance}, \textit{Infection}, and \textit{Action} steps, by which NIDS can detect anomalies. They are called the reconnaissance detector, the infection detector, and the action detector, respectively. Each detector has its classifier learned from networking patterns of the corresponding step. Once a window is given to \sys, each per-step detector takes it as input and determines if it contains any anomalous pattern for the corresponding step. If so, \sys labels the window with the name of the corresponding step. For example, we call a given window a reconnaissance window if the reconnaissance detector detects anomalies from the window. This process provides a precedence relation between windows according to the kill chain steps. 

A window may belong to multiple steps. For example, the window can be classified as \textit{Reconnaissance} and \textit{Infection} by the reconnaissance and the infection detectors. We call such a window both a reconnaissance window and an infection window. As the results of per-step detectors can be false positives, we make our per-step detectors return confidence scores as well as the results. \sys applies the softmax function to normalize the confidence scores from per-step detectors, resulting in a probability distribution. The probability distribution is used to correct false positives by the infection identification algorithm. We call an output of per-step detectors an \textit{event} that contains a window, three labels (\ie whether the window belongs to each step respectively), and four probabilities (\ie normalized confidence scores). 

\ptitle{Sequence analyzer (sequence of events $\rightarrow$ identified infection events).} 
This module runs the infection identification algorithm to find the infection events that lead to the action event. The algorithm takes a sequence of past events, each of which has a probability distribution assigned by per-step detectors. Then, the algorithm analyzes the sequence and determines only one kill chain step for each event according to the entire context. To this end, we develop an identification algorithm based on the attention mechanism in deep learning techniques, which considers all the (hidden) states when producing the next state. Finally, the algorithm returns the infection events from the resulting sequence. 

\ptitle{Detector updater. (identified infection events $\rightarrow$ updated infection detector)}
The detector updater is responsible for updating the classifier of the infection detector. The module labels the identified infection events as \textit{Infection} and re-trains a new classifier with the training set and the events.