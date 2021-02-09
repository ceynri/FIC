<template>
  <section class="compressor page_frame">
    <header class="title_wrapper">
      <h1 class="title">Compressor</h1>
      <div class="comment">Upload facial images to compress</div>
    </header>
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
