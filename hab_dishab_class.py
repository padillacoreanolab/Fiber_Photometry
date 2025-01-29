import numpy as np
import pandas as pd
import tdt
import os
import matplotlib.pyplot as plt

from trial_class import Trial
from experiment_class import Experiment

class HabDishab(Experiment):

    '''********************************** FOR SINGLE OBJECT  **********************************'''

    def hab_dishab_plot_behavior_event(self, behavior_name='all', plot_type='dFF', ax=None):
        """
        Plots Delta F/F (dFF) or z-scored signal with behavior events for the habituation-dishabituation experiment.

        Parameters:
        - behavior_name (str): The name of the behavior to plot. Use 'all' to plot all behaviors.
        - plot_type (str): The type of plot. Options are 'dFF' and 'zscore'.
        - ax: An optional matplotlib Axes object. If provided, the plot will be drawn on this Axes.
        """
        # Prepare data based on plot type
        y_data = []
        if plot_type == 'dFF':
            if self.dFF is None:
                self.compute_dff()
            y_data = self.dFF
            y_label = r'$\Delta$F/F'
            y_title = 'Delta F/F Signal'
        elif plot_type == 'zscore':
            if self.zscore is None:
                self.compute_zscore()
            y_data = self.zscore
            y_label = 'Z-scored ΔF/F'
            y_title = 'Z-scored Signal'
        else:
            raise ValueError("Invalid plot_type. Only 'dFF' and 'zscore' are supported.")

        # Create plot if no axis is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 8))

        ax.plot(self.timestamps, np.array(y_data), linewidth=2, color='black', label=plot_type)

        # Define specific colors for behaviors
        behavior_colors = {'Investigation': 'dodgerblue', 'Approach': 'green', 'Defeat': 'red'}

        # Track which labels have been plotted to avoid duplicates
        plotted_labels = set()

        # Plot behavior spans
        if behavior_name == 'all':
            for behavior_event in self.behaviors.keys():
                if behavior_event in behavior_colors:
                    behavior_onsets = self.behaviors[behavior_event].onset
                    behavior_offsets = self.behaviors[behavior_event].offset
                    color = behavior_colors[behavior_event]

                    for on, off in zip(behavior_onsets, behavior_offsets):
                        label = behavior_event if behavior_event not in plotted_labels else None
                        ax.axvspan(on, off, alpha=0.25, label=label, color=color)
                        plotted_labels.add(behavior_event)
        else:
            # Plot a single behavior
            if behavior_name not in self.behaviors.keys():
                raise ValueError(f"Behavior event '{behavior_name}' not found in behaviors.")
            behavior_onsets = self.behaviors[behavior_name].onset
            behavior_offsets = self.behaviors[behavior_name].offset
            color = behavior_colors.get(behavior_name, 'dodgerblue')  # Default to blue if behavior not in color map

            for on, off in zip(behavior_onsets, behavior_offsets):
                label = behavior_name if behavior_name not in plotted_labels else None
                ax.axvspan(on, off, alpha=0.25, color=color, label=label)
                plotted_labels.add(behavior_name)

        # Plot s1 introduced/removed events if provided
        if hasattr(self, 's1_events') and self.s1_events:
            for on in self.s1_events['introduced']:
                label = 's1 Introduced' if 's1 Introduced' not in plotted_labels else None
                ax.axvline(on, color='blue', linestyle='--', label=label, alpha=0.7)
                plotted_labels.add('s1 Introduced')
            for off in self.s1_events['removed']:
                label = 's1 Removed' if 's1 Removed' not in plotted_labels else None
                ax.axvline(off, color='blue', linestyle='-', label=label, alpha=0.7)
                plotted_labels.add('s1 Removed')

        # Plot s2 introduced/removed events if provided
        if hasattr(self, 's2_events') and self.s2_events:
            for on in self.s2_events['introduced']:
                label = 's2 Introduced' if 's2 Introduced' not in plotted_labels else None
                ax.axvline(on, color='red', linestyle='--', label=label, alpha=0.7)
                plotted_labels.add('s2 Introduced')
            for off in self.s2_events['removed']:
                label = 's2 Removed' if 's2 Removed' not in plotted_labels else None
                ax.axvline(off, color='red', linestyle='-', label=label, alpha=0.7)
                plotted_labels.add('s2 Removed')

        # Add labels and title
        ax.set_ylabel(y_label)
        ax.set_xlabel('Seconds')
        ax.set_title(f'{self.subject_name}: {y_title} with {behavior_name.capitalize()} Bouts' if behavior_name != 'all' else f'{self.subject_name}: {y_title} with All Behavior Events')

        # Manually set more x-tick marks (triple the ticks)
        num_ticks = len(ax.get_xticks()) * 3  # Triples the current number of ticks
        ax.set_xticks(np.linspace(self.timestamps[0], self.timestamps[-1], num_ticks))

        ax.legend()
        plt.tight_layout()

        # Show the plot if no external axis is provided
        if ax is None:
            plt.show()

    def hab_dishab_extract_intruder_bouts(self, csv_base_path):
        """
        Extracts 's1 Introduced', 's1 Removed', 's2 Introduced', and 's2 Removed' events from a CSV file,
        and removes the ITI times (Inter-Trial Intervals) from the data using the remove_time function.

        Parameters:
        - csv_base_path (str): The file path to the CSV file.
        """
        data = pd.read_csv(csv_base_path)

        # Filter rows for specific behaviors
        s1_introduced = data[data['Behavior'] == 's1_Introduced'].head(6)  # Get first 6
        s1_removed = data[data['Behavior'] == 's1_Removed'].head(6)  # Get first 6
        s2_introduced = data[data['Behavior'] == 's2_Introduced']
        s2_removed = data[data['Behavior'] == 's2_Removed']

        # Extract event times
        s1_events = {
            "introduced": s1_introduced['Start (s)'].tolist(),
            "removed": s1_removed['Start (s)'].tolist()
        }

        s2_events = {
            "introduced": s2_introduced['Start (s)'].tolist(),
            "removed": s2_removed['Start (s)'].tolist()
        }

        self.s1_events = s1_events
        self.s2_events = s2_events

        # Now compute z-score with baseline being from initial artifact removal to the first s1 Introduced event
        if s1_events['introduced']:
            baseline_end_time = s1_events['introduced'][0]
            # self.compute_zscore()
            # self.compute_zscore(method='baseline', baseline_start=self.timestamps[0], baseline_end=baseline_end_time)

        # # Remove ITI times (Time between when a mouse is removed and then introduced)
        # for i in range(len(s1_events['removed']) - 1):
        #     s1_removed_time = s1_events['removed'][i]
        #     s1_next_introduced_time = s1_events['introduced'][i + 1]
        #     self.remove_time(s1_removed_time, s1_next_introduced_time)

        # # Handle ITI between last s1 removed and first s2 introduced (if applicable)
        # if s1_events['removed'] and s2_events['introduced']:
        #     self.remove_time(s1_events['removed'][-1], s2_events['introduced'][0])

        # # Handle ITI between s2 removed and the next s2 introduced (for subsequent bouts)
        # for i in range(len(s2_events['removed']) - 1):
        #     s2_removed_time = s2_events['removed'][i]
        #     s2_next_introduced_time = s2_events['introduced'][i + 1]
        #     self.remove_time(s2_removed_time, s2_next_introduced_time)

        # print("ITI times removed successfully.")

    def hab_dishab_find_behavior_events_in_bout(self):
        """
        Finds all behavior events within each bout defined by s1 and s2 introduced and removed. 
        For each event found, returns the start time, end time, total duration, and mean z-score during the event.

        Parameters:
        - s1_events (dict): Dictionary containing "introduced" and "removed" timestamps for s1.
        - s2_events (dict, optional): Dictionary containing "introduced" and "removed" timestamps for s2.

        Returns:
        - bout_dict (dict): Dictionary where each key is the bout number (starting from 1), and the value contains 
                            details about each behavior event found in that bout.
        """
        bout_dict = {}

        # Compute z-score if not already done
        # self.zscore = None
        if self.zscore is None:
            self.compute_zscore()

        # Extract behavior events
        behavior_events = self.behaviors

        # Function to process a bout (sub-function within this function to avoid repetition)
        def process_bout(bout_key, start_time, end_time):
            bout_dict[bout_key] = {}

            # Iterate through each behavior event to find those within the bout
            for behavior_name, behavior_data in behavior_events.items():
                bout_dict[bout_key][behavior_name] = []

                behavior_onsets = np.array(behavior_data.onset)
                behavior_offsets = np.array(behavior_data.offset)

                # Find events that start and end within the bout
                within_bout = (behavior_onsets >= start_time) & (behavior_offsets <= end_time)

                # If any events are found in this bout, process them
                if np.any(within_bout):
                    for onset, offset in zip(behavior_onsets[within_bout], behavior_offsets[within_bout]):
                        # Calculate total duration of the event
                        duration = offset - onset

                        # Find the z-score during this event
                        zscore_indices = (self.timestamps >= onset) & (self.timestamps <= offset)
                        mean_zscore = np.mean(self.zscore[zscore_indices])

                        # Store the details in the dictionary
                        event_dict = {
                            'Start Time': onset,
                            'End Time': offset,
                            'Total Duration': duration,
                            'Mean zscore': mean_zscore
                        }

                        bout_dict[bout_key][behavior_name].append(event_dict)

        # Iterate through each bout defined by s1 introduced and removed
        for i, (start_time, end_time) in enumerate(zip(self.s1_events['introduced'], self.s1_events['removed']), start=1):
            bout_key = f's1_{i}'
            process_bout(bout_key, start_time, end_time)

        for i, (start_time, end_time) in enumerate(zip(self.s2_events['introduced'], self.s2_events['removed']), start=1):
                bout_key = f's2_{i}'
                process_bout(bout_key, start_time, end_time)
            
    
        self.bout_dict = bout_dict

    '''********************************** FOR GROUP CLASS  **********************************'''

    def hab_dishab_processing(self):
        data_rows = []

        for block_folder, tdt_data_obj in self.blocks.items():
            csv_file_name = f"{block_folder}.csv"
            csv_file_path = os.path.join(self.csv_base_path, csv_file_name)

            if os.path.exists(csv_file_path):
                print(f"Hab_Dishab Processing {block_folder}...")

                # Call the three functions in sequence using the CSV file path
                tdt_data_obj.hab_dishab_extract_intruder_bouts(csv_file_path)
                tdt_data_obj.hab_dishab_find_behavior_events_in_bout()
                tdt_data_obj.get_first_behavior()

    '''********************************** PLOTTING  **********************************'''

    def hd_extract_total_behavior_durations(group_data, bouts, behavior='Investigation'):
        """
        Extracts the total durations for the specified behavior (e.g., 'Investigation') 
        for each subject and bout, and returns a DataFrame.

        Parameters:
        group_data (object): The object containing bout data for each subject.
        bouts (list): A list of bout names to process.
        behavior (str): The behavior of interest to calculate total durations for (default is 'Investigation').

        Returns:
        pd.DataFrame: A DataFrame where each row represents a subject, 
                    and each column represents the total duration of the specified behavior for a specific bout.
        """
        # Initialize an empty list to hold the data for each subject
        data_list = []

        # Populate the data_list from the group_data.blocks
        for block_data in group_data.blocks.values():
            if hasattr(block_data, 'bout_dict') and block_data.bout_dict:  # Ensure bout_dict exists and is populated
                # Use the subject name from the TDTData object
                block_data_dict = {'Subject': block_data.subject_name}

                for bout in bouts:  # Only process bouts in the given list of bouts
                    if bout in block_data.bout_dict and behavior in block_data.bout_dict[bout]:
                        # Collect the total duration for the specified behavior for this subject and bout
                        total_duration = np.nansum([event['Total Duration'] for event in block_data.bout_dict[bout][behavior]])
                        block_data_dict[bout] = total_duration
                    else:
                        block_data_dict[bout] = np.nan  # If no data, assign NaN

                # Append the block's data to the data_list
                data_list.append(block_data_dict)

        # Convert the data_list into a DataFrame
        behavior_duration_df = pd.DataFrame(data_list)

        # Set the index to 'Subject'
        behavior_duration_df.set_index('Subject', inplace=True)

        return behavior_duration_df
    
    def hab_dishab_plot_y_across_bouts_gray(df,  title='Mean Across Bouts', ylabel='Mean Value', custom_xtick_labels=None, custom_xtick_colors=None, ylim=None, bar_color='#00B7D7', 
                             yticks_increment=None, xlabel='Intruder',figsize = (12,7), pad_inches = 1):
        """
        Plots the mean values during investigations or other events across bouts with error bars for SEM
        and individual subject lines connecting the bouts. All subjects are plotted in gray.

        Parameters:
        - df (DataFrame): A DataFrame where rows are subjects, and bouts are columns.
                        Values should represent the mean values (e.g., mean DA, investigation times)
                        for each subject and bout.
        - title (str): The title for the plot.
        - ylabel (str): The label for the y-axis.
        - custom_xtick_labels (list): A list of custom x-tick labels. If not provided, defaults to the column names.
        - custom_xtick_colors (list): A list of colors for the x-tick labels. Must be the same length as `custom_xtick_labels`.
        - ylim (tuple): A tuple (min, max) to set the y-axis limits. If None, the limits are set automatically based on the data.
        - bar_color (str): The color to use for the bars (default is cyan).
        - yticks_increment (float): Increment amount for the y-axis ticks.
        - xlabel (str): The label for the x-axis.
        """

        # Calculate the mean and SEM for each bout (across all subjects)
        mean_values = df.mean()
        sem_values = df.sem()

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)  #12,7

        # Plot the bar plot with error bars (mean and SEM) without adding it to the legend
        bars = ax.bar(
            df.columns, 
            mean_values, 
            yerr=sem_values, 
            capsize=6,  # Increase capsize for larger error bars
            color=bar_color,  # Customizable bar color
            edgecolor='black', 
            linewidth=4,  # Thicker and darker bar outlines
            width=0.6,
            error_kw=dict(elinewidth=4, capthick=4,capsize=10,zorder=5)  # Thicker error bars and make them appear above circles
            # elinewidth = 2.5, capthick = 2.5
        )

        # Plot all subject lines and markers in gray
        for i, subject in enumerate(df.index):
            ax.plot(df.columns, df.loc[subject], linestyle='-', color='gray', alpha=0.5, linewidth=2.5, zorder=1)

        # Plot unfilled circle markers with larger size, in gray
        for i, subject in enumerate(df.index):
            ax.scatter(df.columns, df.loc[subject], facecolors='none', edgecolors='gray', s=120, alpha=0.6, linewidth=4, zorder=2)

        # Add labels, title, and format
        ax.set_ylabel(ylabel, fontsize=28, labelpad=12)  # Larger y-axis label
        ax.set_xlabel(xlabel, fontsize=40, labelpad=12)
        # ax.set_title(title, fontsize=16)

        # Set x-ticks to match the bout labels
        ax.set_xticks(np.arange(len(df.columns)))

        # Use custom x-tick labels if provided, otherwise use the column names
        if custom_xtick_labels is not None:
            ax.set_xticklabels(custom_xtick_labels, fontsize=28)
            if custom_xtick_colors is not None:
                for tick, color in zip(ax.get_xticklabels(), custom_xtick_colors):
                    tick.set_color(color)
        else:
            ax.set_xticklabels(df.columns, fontsize=26)

        # Increase the font size of y-axis tick numbers
        ax.tick_params(axis='y', labelsize=38)  # Increase y-axis number size
        ax.tick_params(axis='x', labelsize=38)  # Optional: also increase x-axis number size

        # Automatically set the y-limits based on the data range if ylim is not provided
        if ylim is None:
            # Collect all values to determine the y-limits
            all_values = np.concatenate([df.values.flatten(), mean_values.values.flatten()])
            min_val = np.nanmin(all_values)
            max_val = np.nanmax(all_values)

            # Set lower y-limit to 0 if all values are above 0, otherwise set to the minimum value
            lower_ylim = 0 if min_val > 0 else min_val * 1.1
            upper_ylim = max_val * 1.1  # Adding a bit of space above the highest value
            
            ax.set_ylim(lower_ylim, upper_ylim)
        else:
            # If ylim is provided, set the limits to the specified values
            ax.set_ylim(ylim)
            if ylim[0] < 0:
                ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)

        # Set y-ticks based on yticks_increment
        if yticks_increment is not None:
            y_min, y_max = ax.get_ylim()
            y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
            ax.set_yticks(y_ticks)

        # Remove the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(5)    # Left axis line
        ax.spines['bottom'].set_linewidth(5)  # Bottom axis line


        plt.savefig(f'{title}{ylabel[0]}.png', transparent=True, bbox_inches='tight', pad_inches=pad_inches)
        # Display the plot without legend
        plt.tight_layout()
        plt.show()