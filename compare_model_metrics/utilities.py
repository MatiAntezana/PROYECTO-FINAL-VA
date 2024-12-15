import numpy as np
import matplotlib.pyplot as plt

def show_imgs_in_RGB(imgs: list, cols: int = 1, size: tuple = (12, 10), img_names: list = None, name_size: int = None, save: bool = False):
    """
    Show multiple images using matplotlib in a single plot.

    Parameters
    ----------
    imgs : list of np.ndarray
        List of images to show.
    cols : int, optional
        Number of columns in the plot, by default 1.
    size : tuple, optional
        Size of the figure, by default (12, 10).
    img_names : list of str, optional
        List of image names, by default None.
    name_size : int, optional
        Font size of the names, by default None.
    save : bool, optional
        Save the plot as a .png file, by default False.

    Returns
    -------
    None
    """
    if not all(isinstance(img, np.ndarray) for img in imgs): raise TypeError("All images must be numpy arrays")
    if type(size) != tuple or len(size) != 2: raise ValueError("size must be a tuple with 2 elements")
    if size[0] <= 0 or size[1] <= 0: raise ValueError("size elements must be greater than 0")
    if not isinstance(cols, int) or cols <= 0: raise ValueError("cols must be a positive integer")

    rows = (len(imgs) + cols - 1) // cols

    plt.figure(figsize=size)

    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
        if img_names and len(img_names) > i:
            plt.title(img_names[i], fontsize=name_size if name_size else None)

    plt.tight_layout()

    if save: plt.savefig("plots/Images.pdf", dpi=600, bbox_inches='tight')

    plt.show()


def F1_score(precision: float, recall: float) -> float:
    """
    Calculate the F1 score given the precision and recall values.

    Parameters:
    ----------
    precision : float
        The precision value.
    recall : float
        The recall value.

    Returns:
    -------
    float
        The F1 score value.
    """
    return 2 * (precision * recall) / (precision + recall)


def metrics_averaging(metrics: list) -> np.ndarray:
    """
    Calculate the mean of the metrics.

    Parameters:
    ----------
    metrics : list
        The list of metrics.

    Returns:
    -------
    np.ndarray
        The mean of the metrics.
    """
    all_metrics = np.array(metrics)  
    mean_metrics = all_metrics.mean(axis=0) 
    return mean_metrics


def plot_metrics(metrics_dict: dict, labels: list, colors: list = None, figsize: tuple = (14 ,8), fontsize: int = 16, ticksize: int = 12, title: bool = False, titlesize: int = 22, save: bool = False) -> None:
    """	
    Plot a comparison of metrics for different models.

    Parameters
    ----------
    metrics_dict : dict
        The dictionary with the metrics values.
    labels : list
        The labels of the models.
    colors : list, optional
        The colors of the bars, by default None.
    figsize : tuple, optional
        The size of the figure, by default (14, 8).
    fontsize : int, optional
        The font size, by default 16.
    ticksize : int, optional
        The size of the ticks, by default 12.
    title : bool, optional
        Show the title, by default False.
    titlesize : int, optional
        The size of the title, by default 22.
    save : bool, optional
        Save the plot, by default False.

    Returns
    -------
    None
    """
    for metric_values in metrics_dict.values():
        if len(metric_values) != len(labels):
            raise ValueError("All metrics must have the same length as the labels.")
    
    num_metrics = len(metrics_dict)  
    
    if colors is None:
        colors = plt.cm.tab10.colors[:num_metrics]
    elif len(colors) != num_metrics:
        raise ValueError("The number of colors must be the same as the number of metrics.")

    x = np.arange(len(labels)) 
    width = 0.8 / num_metrics  
    offset = -width * (num_metrics - 1) / 2  
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (metric_name, metric_values) in enumerate(metrics_dict.items()):
        ax.bar(x + offset + i * width, metric_values, width, label=metric_name, color=colors[i])

        for j in range(len(metric_values)):
            ax.text(x[j] + offset + i * width, metric_values[j] + 0.02, f'{metric_values[j]:.2f}', 
                    ha='center', fontsize=10)
    

    ax.set_xlabel(r'Models', fontsize=fontsize)
    ax.set_ylabel(r'Values', fontsize=fontsize)
    if title: ax.set_title(r'Metrics Comparison', fontsize=titlesize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=ticksize, frameon=False)  
    plt.tight_layout()

    if save: plt.savefig("plots/Metrics Comparison.pdf", dpi=600, bbox_inches='tight')

    plt.show()


def recall_confidence_curve(confidence: np.ndarray, recall: np.ndarray, labels: list, color: list, figsize: tuple = (14, 10), fontsize: int = 20, ticksize: int = 12, title: bool = False, titlesize: int = 26, save: bool = False) -> None:
    """
    Plot the confidence-recall curve.

    Parameters
    ----------
    confidence : np.ndarray
        The confidence values.
    recall : np.ndarray
        The recall values.
    labels : list
        The labels of the curves.
    color : list
        The colors of the curves.
    figsize : tuple
        The size of the figure. The default is (14, 8).
    fontsize : int, optional
        The font size, by default 20.
    ticksize : int, optional
        The size of the ticks, by default 14.
    title : bool, optional
        Show the title, by default False.
    titlesize : int, optional
        The size of the title, by default 26.   
    save : bool, optional
        Save the plot, by default False.

    Returns
    -------
    None
    """
    if not all(isinstance(conf, np.ndarray) for conf in confidence): raise TypeError("All confidences must be lists")	
    if not all(isinstance(rec, np.ndarray) for rec in recall): raise TypeError("All recalls must be lists")
    if not all(isinstance(label, str) for label in labels): raise TypeError("All labels must be strings")
    if not all(isinstance(col, str) for col in color): raise TypeError("All colors must be strings")
    if type(figsize) != tuple or len(figsize) != 2: raise ValueError("figsize must be a tuple with 2 elements")
    if figsize[0] <= 0 or figsize[1] <= 0: raise ValueError("figsize elements must be greater than 0")
    if not isinstance(fontsize, int) or fontsize <= 0: raise ValueError("fontsize must be a positive integer")
    if not isinstance(ticksize, int) or ticksize <= 0: raise ValueError("ticksize must be a positive integer")
    if not isinstance(title, bool): raise TypeError("title must be a boolean")
    if not isinstance(titlesize, int) or titlesize <= 0: raise ValueError("titlesize must be a positive integer") 
    if not isinstance(save, bool): raise TypeError("save must be a boolean")

    plt.figure(figsize=figsize)
    for i in range(len(recall)):
        plt.plot(confidence[i], recall[i], label=labels[i], color=color[i])

    plt.ylabel(r'$Recall = \frac{TP}{TP + FN}$', fontsize=fontsize)
    plt.xlabel(r"$Confidence = P(object) \cdot IoU$", fontsize=fontsize)
    ticks_x = np.arange(0, 1.1, 0.1)  
    plt.xticks(ticks_x, fontsize=ticksize)	
    plt.yticks(ticks_x, fontsize=ticksize)
    plt.xlim(0, 1.005)
    plt.ylim(-0.005, 1.005)
    
    if (title): plt.title("Recall-Confidence Curve", fontsize=titlesize)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=ticksize, frameon=False)
    plt.grid(True)

    if (save): plt.savefig("plots/Recall-Confidence Curve.pdf", dpi=600, bbox_inches='tight')

    plt.show()


def F1_confidence_curve(confidence: np.ndarray, F1_score: np.ndarray, labels: list, color: list, figsize: tuple = (14 ,10), fontsize: int = 20, ticksize: int = 12, title: bool = False, titlesize: int = 26, save: bool = False) -> None:
    """
    Plot the confidence-F1 score curve.

    Parameters
    ----------
    confidence : np.ndarray
        The confidence values.
    F1_score : np.ndarray
        The F1 score values.
    labels : list
        The labels of the curves.
    color : list
        The colors of the curves.
    figsize : tuple
        The size of the figure. The default is (14, 8).
    fontsize : int, optional
        The font size, by default 20.
    ticksize : int, optional
        The size of the ticks, by default 14.
    title : bool, optional
        Show the title, by default False.
    titlesize : int, optional
        The size of the title, by default 26.
    save : bool, optional
        Save the plot, by default False.

    Returns
    -------
    None
    """
    if not all(isinstance(conf, np.ndarray) for conf in confidence): raise TypeError("All confidences must be lists")	
    if not all(isinstance(F1, np.ndarray) for F1 in F1_score): raise TypeError("All F1 scores must be lists")
    if not all(isinstance(label, str) for label in labels): raise TypeError("All labels must be strings")
    if not all(isinstance(col, str) for col in color): raise TypeError("All colors must be strings")
    if type(figsize) != tuple or len(figsize) != 2: raise ValueError("figsize must be a tuple with 2 elements")
    if figsize[0] <= 0 or figsize[1] <= 0: raise ValueError("figsize elements must be greater than 0")
    if not isinstance(fontsize, int) or fontsize <= 0: raise ValueError("fontsize must be a positive integer")
    if not isinstance(ticksize, int) or ticksize <= 0: raise ValueError("ticksize must be a positive integer")
    if not isinstance(title, bool): raise TypeError("title must be a boolean")
    if not isinstance(titlesize, int) or titlesize <= 0: raise ValueError("titlesize must be a positive integer") 
    if not isinstance(save, bool): raise TypeError("save must be a boolean")

    plt.figure(figsize=figsize)
    for i in range(len(F1_score)):
        plt.plot(confidence[i], F1_score[i], label=labels[i], color=color[i])

    plt.ylabel(r'$F1 Score = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$', fontsize=fontsize)
    plt.xlabel(r"$Confidence = P(object) \cdot IoU$", fontsize=fontsize)
    ticks_x = np.arange(0, 1.1, 0.1)  
    plt.xticks(ticks_x, fontsize=ticksize)	
    plt.yticks(ticks_x, fontsize=ticksize)
    plt.xlim(0, 1.005)
    plt.ylim(-0.005, 1.005)
    
    if (title): plt.title("F1-Confidence Curve", fontsize=titlesize)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=ticksize, frameon=False)
    plt.grid(True)

    if (save): plt.savefig("plots/F1-Confidence Curve.pdf", dpi=600, bbox_inches='tight')

    plt.show()


def precision_confidence_curve(confidence: np.ndarray, precision: np.ndarray, labels: list, color: list, figsize: tuple = (14 ,10), fontsize: int = 20, ticksize: int = 12, title: bool = False, titlesize: int = 26, save: bool = False) -> None:
    """
    Plot the confidence-precision curve.

    Parameters
    ----------
    confidence : np.ndarray
        The confidence values.
    precision : np.ndarray
        The precision values.
    labels : list
        The labels of the curves.
    color : list
        The colors of the curves.
    figsize : tuple
        The size of the figure. The default is (14, 8).
    fontsize : int, optional
        The font size, by default 20.
    ticksize : int, optional
        The size of the ticks, by default 14.
    title : bool, optional
        Show the title, by default False.
    titlesize : int, optional
        The size of the title, by default 26.
    save : bool, optional
        Save the plot, by default False.

    Returns
    -------
    None
    """
    if not all(isinstance(conf, np.ndarray) for conf in confidence): raise TypeError("All confidences must be lists")	
    if not all(isinstance(pre, np.ndarray) for pre in precision): raise TypeError("All precisions must be lists")
    if not all(isinstance(label, str) for label in labels): raise TypeError("All labels must be strings")
    if not all(isinstance(col, str) for col in color): raise TypeError("All colors must be strings")
    if type(figsize) != tuple or len(figsize) != 2: raise ValueError("figsize must be a tuple with 2 elements")
    if figsize[0] <= 0 or figsize[1] <= 0: raise ValueError("figsize elements must be greater than 0")
    if not isinstance(fontsize, int) or fontsize <= 0: raise ValueError("fontsize must be a positive integer")
    if not isinstance(ticksize, int) or ticksize <= 0: raise ValueError("ticksize must be a positive integer")
    if not isinstance(title, bool): raise TypeError("title must be a boolean")
    if not isinstance(titlesize, int) or titlesize <= 0: raise ValueError("titlesize must be a positive integer") 
    if not isinstance(save, bool): raise TypeError("save must be a boolean")

    plt.figure(figsize=figsize)
    for i in range(len(precision)):
        plt.plot(confidence[i], precision[i], label=labels[i], color=color[i])

    plt.ylabel(r'$Precision = \frac{TP}{TP + FP}$', fontsize=fontsize)
    plt.xlabel(r"$Confidence = P(object) \cdot IoU$", fontsize=fontsize)   
    ticks_x = np.arange(0, 1.1, 0.1)
    plt.xticks(ticks_x, fontsize=ticksize)	
    plt.yticks(ticks_x, fontsize=ticksize)
    plt.xlim(0, 1.005)
    plt.ylim(0, 1.005)
    
    if (title): plt.title("Precision-Confidence Curve", fontsize=titlesize)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=ticksize, frameon=False)
    plt.grid(True)

    if (save): plt.savefig("plots/Precision-Confidence Curve.pdf", dpi=600, bbox_inches='tight')

    plt.show()


def precision_recall_curve(recall: np.ndarray, precision: np.ndarray, labels: list, color: list, figsize: tuple = (14 ,10), fontsize: int = 20, ticksize: int = 12, title: bool = False, titlesize: int = 26, save: bool = False) -> None:
    """	
    Plot the precision-recall curve.

    Parameters
    ----------
    recall : np.ndarray
        The recall values.
    precision : np.ndarray
        The precision values.
    labels : list
        The labels of the curves.
    color : list
        The colors of the curves.
    figsize : tuple
        The size of the figure. The default is (14, 8).
    fontsize : int, optional
        The font size, by default 20.
    ticksize : int, optional
        The size of the ticks, by default 14.
    title : bool, optional
        Show the title, by default False.
    titlesize : int, optional
        The size of the title, by default 26.
    save : bool, optional
        Save the plot, by default False.

    Returns
    -------
    None
    """
    if not all(isinstance(rec, np.ndarray) for rec in recall): raise TypeError("All recalls must be lists")	
    if not all(isinstance(pre, np.ndarray) for pre in precision): raise TypeError("All precisions must be lists")
    if not all(isinstance(label, str) for label in labels): raise TypeError("All labels must be strings")
    if not all(isinstance(col, str) for col in color): raise TypeError("All colors must be strings")
    if type(figsize) != tuple or len(figsize) != 2: raise ValueError("figsize must be a tuple with 2 elements")
    if figsize[0] <= 0 or figsize[1] <= 0: raise ValueError("figsize elements must be greater than 0")
    if not isinstance(fontsize, int) or fontsize <= 0: raise ValueError("fontsize must be a positive integer")
    if not isinstance(ticksize, int) or ticksize <= 0: raise ValueError("ticksize must be a positive integer")
    if not isinstance(title, bool): raise TypeError("title must be a boolean")
    if not isinstance(titlesize, int) or titlesize <= 0: raise ValueError("titlesize must be a positive integer") 
    if not isinstance(save, bool): raise TypeError("save must be a boolean")

    plt.figure(figsize=figsize)
    for i in range(len(precision)):
        plt.plot(recall[i], precision[i], label=labels[i], color=color[i])

    plt.ylabel(r'$Precision = \frac{TP}{TP + FP}$', fontsize=fontsize)
    plt.xlabel(r"$Recall = \frac{TP}{TP + FN}$", fontsize=fontsize)
    ticks_x = np.arange(0, 1.1, 0.1)  
    plt.xticks(ticks_x, fontsize=ticksize)	
    plt.yticks(ticks_x, fontsize=ticksize)
    plt.xlim(0, 1.005)
    plt.ylim(0, 1.010)
    
    if (title): plt.title("Precision-Recall Curve", fontsize=titlesize)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=ticksize, frameon=False)
    plt.grid(True)

    if (save): plt.savefig("plots/Precision-Recall Curve.pdf", dpi=600, bbox_inches='tight')
    
    plt.show()