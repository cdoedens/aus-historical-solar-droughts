# Australian Solar Droughts
Identifying periods of low solar irradiance based on satellite irradiance data

This project aims to fill a gap in existing solar pv research regarding the presence and character of solar droughts (i.e. dunkelflaute) in Australia. Previous research into resources for solar pv generation have focussed on total availability of energy (i.e. capacity factor) and other long-term parameters. The research reflects the existing states of most energy markets, where variable renewable energy (VRE) sources such as solar pv only make up a small part of the total grid capacity, and therefore energy demand during a period of low solar irradiance can be met by dispatchable energy sources (i.e. coal and gas). However, as VRE increases and the market relies on it more and more to meet demand, a period of low irradiance can be highly consequential. Therefore, understanding the character of these "solar droughts" is vital to maintain long-term system reliability.

Dunkelflaute is a Germen word that translates to "dark doldrums" or "dark lull". It has become a common term used to refer to periods with low VRE potential, i.e. the sun isn't shining and the wind isn't blowing.

The main goals of this project is to determine:
- Where are the regions that experience the most solar drought?
- When are the periods that have the most solar droughts?
- How long are the droughts in different regions and periods?
- How severe are the droughts in different regions and periods?
- How much solar drought should the energy system prepare for?

The first four questions are considered to be the important defining characteristics of solar droughts, and combine to become the important criteria for answering the final question.

## Solar drought definition
The first problem that needs addressing in a project like this is how a "solar drought" should be defined. There are many possible definitions that could be used, each with their own strengths and weaknesses. For example, you could a statistical definition, where the lowest 10th percentile of solar irradiance for a particular region is classified as a drought. This definition takes into account both the mean and the variability of the irradiance in a region, which is potentially ideal for the end user, but it also means that all regions will have the same number of droughts, potentially not an ideal feature. Another definition could be proportional to a region's mean or max, so that any irradiance that falls below a given proportion is classified as a drought. This definition takes into account regional variations in irradiance, a useful feature that allows comparison between latitudes (e.g. Hobart and Darwin), but can be sensitive to the choice of threshold proportion. A completely different approach could include energy demand data or estimates, and define a drought as a mismatch between supply (irradiance) and demand. 

Another important feature of the definition is the choice of timeframe for the drought. The definition could apply to daily average irradiance, or it could incorporate hourly or sub-hourly data to identify important periods of the day where irradiance falls below a given threshold.

Ultimately, as the purpose of this project is to guide energy systems planning, the choice of definition is best guided by the system planners and energy market operators.

At the moment, the project is using a "mean proportional" and "max proportional" definition due to the ability to easilty rotate between them. 

## TO DO
 - Explain observed solar droughts by identifying relevant meteorological systems and atmospheric variables
 - Investigate specific case studies in more depth
 - Determine conclusions that can be drawn
 - Decide on journal for publication

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries, reach out to [cdoedens@student.unimelb.edu.au](mailto:cdoedens@student.unimelb.edu.au)
