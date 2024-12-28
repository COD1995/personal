## Personal Website

This website is built using the [Al-Folio template](https://github.com/alshedivat/al-folio), a versatile Jekyll-based theme tailored for academic and personal sites. While the original template offers excellent features, I have made several custom modifications to suit my preferences and requirements.

### Key Customizations

1. **Slide Presentation Layout**  
   I have customized the slide presentation styles to better align with my needs. These modifications are primarily implemented in the following files:
   - **`slide.liquid`**: Handles the layout and rendering of slides.
   - **`_base.scss`**: Contains the core styling changes for slides and other components.  
     
     _Note:_ For some unexplained reason, styles defined in **`slide.scss`** are not applied correctly. To avoid this issue, I chose to work within **`_base.scss`** instead. If you happen to identify the root cause of this behavior, **I would greatly appreciate your insights**.

2. **Additional Style Adjustments**  
   Many other custom styles and tweaks have been incorporated into **`_base.scss`** to enhance the overall aesthetics and functionality of the site. These include minor visual refinements, layout adjustments, and custom components.

3. **Numbered TOC**

   Sadly, the original template does not support a numbered table of contents (TOC). To address this limitation, I have added a custom script to automatically number the TOC items. This script is included in **`_layouts/default.liquid`**.


