<template>
  <section class="compressor page_frame">
    <div class="header_wrapper">
      <header class="title_wrapper">
        <h1 class="title">Compress</h1>
        <div class="comment">Upload facial images to compress</div>
      </header>
      <!-- <button class="action_btn">Download All</button> -->
    </div>
    <Uploader accept="image/*" />
  </section>
</template>

<script>
import Uploader from '@/components/Uploader.vue';

export default {
  components: {
    Uploader,
  },
};
</script>

<style lang="scss" scoped>
.compressor {
}
</style>
